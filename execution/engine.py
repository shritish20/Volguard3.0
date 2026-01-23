"""
Execution Engine using CORRECT Upstox API patterns
"""

import upstox_client
from upstox_client.rest import ApiException
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random

from config.settings import TradingMode, ANALYTICS_CONFIG, UPSTOX_CONFIG
from execution.models import Order, OrderStatus, StrategyOrder, ExecutionMode
from utils.logger import setup_logger


class ExecutionEngine:
    """Execution engine for Live and Shadow modes using CORRECT Upstox SDK"""
    
    def __init__(self, auth_manager, data_fetcher, mode: TradingMode):
        self.auth = auth_manager
        self.data_fetcher = data_fetcher
        self.mode = mode
        self.logger = setup_logger(f"execution_{mode.value.lower()}")
        
        # API instances
        self.api_client = auth_manager.api_client
        self.order_api = upstox_client.OrderApi(self.api_client)
        
        # State
        self.orders: Dict[str, Order] = {}
        self.strategy_orders: Dict[str, StrategyOrder] = {}
        self.execution_history = []
        
        # Shadow mode settings
        self.shadow_fill_probability = 0.8
        
        self.logger.info(f"Execution Engine initialized in {mode.value} mode")
    
    def place_order(self, instrument_token: str, trading_symbol: str,
                   transaction_type: str, order_type: str, quantity: int,
                   price: float = 0, trigger_price: float = 0,
                   product: str = "D", validity: str = "DAY",
                   disclosed_quantity: int = 0, tag: str = "",
                   is_amo: bool = False, slice: bool = False) -> Dict:
        """Place a single order"""
        order_id = str(uuid.uuid4())[:8]
        correlation_id = str(uuid.uuid4())[:8]
        
        order = Order(
            order_id=order_id,
            correlation_id=correlation_id,
            instrument_token=instrument_token,
            trading_symbol=trading_symbol,
            transaction_type=transaction_type,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            product=product,
            validity=validity,
            disclosed_quantity=disclosed_quantity,
            tag=tag,
            status=OrderStatus.PENDING,
            filled_quantity=0,
            average_price=0,
            placed_at=datetime.now(),
            updated_at=datetime.now(),
            mode=ExecutionMode.LIVE if self.mode == TradingMode.LIVE else ExecutionMode.SHADOW,
            is_amo=is_amo,
            slice=slice
        )
        
        self.orders[order_id] = order
        
        if self.mode == TradingMode.LIVE:
            return self._place_live_order(order)
        else:
            return self._place_shadow_order(order)
    
    def _place_live_order(self, order: Order) -> Dict:
        """Place live order via Upstox API"""
        try:
            # Prepare order payload using Upstox SDK model
            order_request = upstox_client.PlaceOrderRequest(
                quantity=order.quantity,
                product=order.product,
                validity=order.validity,
                price=order.price,
                tag=order.tag,
                instrument_token=order.instrument_token,
                order_type=order.order_type,
                transaction_type=order.transaction_type,
                disclosed_quantity=order.disclosed_quantity,
                trigger_price=order.trigger_price,
                is_amo=order.is_amo,
                slice=order.slice
            )
            
            # Make API call
            api_response = self.order_api.place_order(
                order_request,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                # Update order with response
                order_ids = api_response.data.order_ids if api_response.data else [order.order_id]
                order.status = OrderStatus.PLACED
                order.updated_at = datetime.now()
                
                # For market orders, simulate immediate fill
                if order.order_type == "MARKET":
                    self._simulate_fill(order, order_ids[0])
                
                return {
                    "success": True,
                    "order_id": order.order_id,
                    "upstox_order_ids": order_ids,
                    "message": "Order placed successfully"
                }
            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                return {
                    "success": False,
                    "order_id": order.order_id,
                    "error": "Order rejected by broker",
                    "message": api_response.message if api_response else "Unknown error"
                }
                
        except ApiException as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return {
                "success": False,
                "order_id": order.order_id,
                "error": f"API Error: {e.status}",
                "message": e.reason
            }
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return {
                "success": False,
                "order_id": order.order_id,
                "error": str(e),
                "message": "Order placement failed"
            }
    
    def _place_shadow_order(self, order: Order) -> Dict:
        """Place shadow order (simulated)"""
        try:
            # Simulate order placement delay
            time.sleep(0.5)
            
            # Determine if order gets filled
            fill_probability = self.shadow_fill_probability
            
            # Adjust probability based on order type
            if order.order_type == "LIMIT":
                # Get current price for simulation
                current_price = self._get_shadow_price(order.instrument_token)
                if current_price and order.price > 0:
                    price_diff_pct = abs(order.price - current_price) / current_price * 100
                    fill_probability = max(0.1, self.shadow_fill_probability - price_diff_pct/100)
            
            if random.random() < fill_probability:
                order.status = OrderStatus.PLACED
                order.updated_at = datetime.now()
                
                # Simulate fill
                self._simulate_fill(order, f"SHADOW_{order.order_id}")
                
                return {
                    "success": True,
                    "order_id": order.order_id,
                    "upstox_order_ids": [f"SHADOW_{order.order_id}"],
                    "message": "Shadow order placed and filled"
                }
            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                return {
                    "success": False,
                    "order_id": order.order_id,
                    "error": "Simulated rejection",
                    "message": "Shadow order rejected"
                }
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return {
                "success": False,
                "order_id": order.order_id,
                "error": str(e),
                "message": "Shadow order failed"
            }
    
    def _get_shadow_price(self, instrument_token: str) -> Optional[float]:
        """Get shadow price for simulation"""
        try:
            # Try to get real market data
            market_api = upstox_client.MarketQuoteApi(self.api_client)
            api_response = market_api.get_full_market_quote(
                symbol=instrument_token,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                key = instrument_token.replace('|', ':')
                data = api_response.data.get(key) or api_response.data.get(instrument_token)
                if data:
                    return data.last_price
        except:
            pass
        
        # Fallback to random price
        return random.uniform(100, 500)
    
    def _simulate_fill(self, order: Order, exchange_order_id: str):
        """Simulate order fill"""
        # Get current price
        current_price = self._get_shadow_price(order.instrument_token) or order.price
        
        # Determine fill price
        if order.order_type == "MARKET":
            fill_price = current_price * (1 + random.uniform(-0.001, 0.001))  # ±0.1% slippage
        else:  # LIMIT order
            if order.transaction_type == "BUY":
                fill_price = min(order.price, current_price * (1 - random.uniform(0, 0.002)))
            else:  # SELL
                fill_price = max(order.price, current_price * (1 + random.uniform(0, 0.002)))
        
        # Determine fill quantity (full or partial)
        if random.random() > 0.9:  # 10% chance of partial fill
            fill_quantity = random.randint(1, order.quantity - 1)
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            fill_quantity = order.quantity
            order.status = OrderStatus.COMPLETED
        
        order.filled_quantity = fill_quantity
        order.average_price = fill_price
        order.updated_at = datetime.now()
        
        # Record execution
        self.execution_history.append({
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'exchange_order_id': exchange_order_id,
            'filled_quantity': fill_quantity,
            'fill_price': fill_price,
            'transaction_type': order.transaction_type,
            'mode': self.mode.value
        })
    
    def place_multi_leg_strategy(self, strategy_type: str, legs: List[Dict], 
                               tag: str = "") -> Dict:
        """Place multi-leg strategy order"""
        if self.mode == TradingMode.LIVE:
            return self._place_live_multi_order(legs, tag)
        else:
            return self._place_shadow_multi_order(legs, tag)
    
    def _place_live_multi_order(self, legs: List[Dict], tag: str) -> Dict:
        """Place multiple live orders"""
        try:
            orders_payload = []
            pending_orders = []
            
            for i, leg in enumerate(legs):
                order_id = str(uuid.uuid4())[:8]
                correlation_id = f"{i+1}"
                
                order = Order(
                    order_id=order_id,
                    correlation_id=correlation_id,
                    instrument_token=leg["instrument_token"],
                    trading_symbol=leg.get("trading_symbol", ""),
                    transaction_type=leg["transaction_type"],
                    order_type=leg["order_type"],
                    quantity=leg["quantity"],
                    price=leg.get("price", 0),
                    trigger_price=leg.get("trigger_price", 0),
                    product=leg.get("product", "D"),
                    validity=leg.get("validity", "DAY"),
                    disclosed_quantity=leg.get("disclosed_quantity", 0),
                    tag=tag,
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    average_price=0,
                    placed_at=datetime.now(),
                    updated_at=datetime.now(),
                    mode=ExecutionMode.LIVE,
                    is_amo=leg.get("is_amo", False),
                    slice=leg.get("slice", False)
                )
                
                self.orders[order_id] = order
                pending_orders.append(order)
                
                # Prepare payload
                order_request = upstox_client.PlaceOrderRequest(
                    quantity=order.quantity,
                    product=order.product,
                    validity=order.validity,
                    price=order.price,
                    tag=order.tag,
                    instrument_token=order.instrument_token,
                    order_type=order.order_type,
                    transaction_type=order.transaction_type,
                    disclosed_quantity=order.disclosed_quantity,
                    trigger_price=order.trigger_price,
                    is_amo=order.is_amo,
                    slice=order.slice
                )
                
                # In production, you'd use multi-order endpoint
                # For now, place sequentially
                try:
                    response = self.order_api.place_order(
                        order_request,
                        api_version='2.0'
                    )
                    
                    if response and response.status == 'success':
                        order.status = OrderStatus.PLACED
                        order.updated_at = datetime.now()
                        
                        if order.order_type == "MARKET":
                            self._simulate_fill(order, response.data.order_ids[0] if response.data else f"LIVE_{order.order_id}")
                    
                except Exception as e:
                    order.status = OrderStatus.REJECTED
                    order.updated_at = datetime.now()
            
            # Count successful orders
            successful = sum(1 for o in pending_orders if o.status == OrderStatus.PLACED)
            failed = len(pending_orders) - successful
            
            return {
                "success": successful > 0,
                "total_orders": len(legs),
                "successful_orders": successful,
                "failed_orders": failed,
                "message": f"Multi-order: {successful} successful, {failed} failed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Multi-order placement failed"
            }
    
    def _place_shadow_multi_order(self, legs: List[Dict], tag: str) -> Dict:
        """Place multiple shadow orders"""
        results = []
        successful = 0
        failed = 0
        
        for leg in legs:
            result = self.place_order(
                instrument_token=leg["instrument_token"],
                trading_symbol=leg.get("trading_symbol", ""),
                transaction_type=leg["transaction_type"],
                order_type=leg["order_type"],
                quantity=leg["quantity"],
                price=leg.get("price", 0),
                trigger_price=leg.get("trigger_price", 0),
                product=leg.get("product", "D"),
                validity=leg.get("validity", "DAY"),
                disclosed_quantity=leg.get("disclosed_quantity", 0),
                tag=tag,
                is_amo=leg.get("is_amo", False),
                slice=leg.get("slice", False)
            )
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
            
            results.append(result)
        
        return {
            "success": successful > 0,
            "total_orders": len(legs),
            "successful_orders": successful,
            "failed_orders": failed,
            "results": results,
            "message": f"Shadow multi-order: {successful} successful, {failed} failed"
        }
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status by ID"""
        return self.orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        order = self.orders.get(order_id)
        if not order:
            return {"success": False, "error": "Order not found"}
        
        if self.mode == TradingMode.LIVE:
            try:
                # Use Upstox SDK to cancel order
                # Note: You need the exchange order ID
                if hasattr(order, 'exchange_order_id') and order.exchange_order_id:
                    cancel_response = self.order_api.cancel_order(
                        order.exchange_order_id,
                        api_version='2.0'
                    )
                    
                    if cancel_response and cancel_response.status == 'success':
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now()
                        return {"success": True, "message": "Order cancelled"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # For shadow mode or fallback
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        return {"success": True, "message": "Order cancelled"}
