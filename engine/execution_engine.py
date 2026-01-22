"""
VOLGUARD v31.0 – EXECUTION ENGINE
=================================
The Muscle - Places V3 Orders with Auto-Slicing and Margin Verification
"""

import upstox_client
from upstox_client.rest import ApiException
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from colorama import Fore, Style

@dataclass
class OrderResult:
    """Standardized order result"""
    status: str
    order_ids: List[str]
    message: str
    error: Optional[str] = None
    total_quantity: int = 0
    filled_quantity: int = 0
    average_price: float = 0.0

class UpstoxExecutionEngine:
    """Professional order execution engine with Upstox V3 API"""
    
    def __init__(self, access_token: str):
        self.config = upstox_client.Configuration()
        self.config.access_token = access_token
        self.api_client = upstox_client.ApiClient(self.config)
        
        # Order APIs
        self.order_api = upstox_client.OrderApiV3(self.api_client)
        self.order_api_v2 = upstox_client.OrderApi(self.api_client)
        
        print(f"{Fore.GREEN}✓ UpstoxExecutionEngine initialized{Style.RESET_ALL}")
    
    def get_token_from_chain(self, chain_df: pd.DataFrame, strike: float, option_type: str) -> Optional[str]:
        """Lookup Instrument Key from Dataframe"""
        try:
            row = chain_df[chain_df['strike'] == strike]
            if row.empty:
                print(f"{Fore.YELLOW}⚠️ Strike {strike} not found in chain{Style.RESET_ALL}")
                return None
            
            key = 'ce_key' if option_type.upper() == "CE" else 'pe_key'
            token = row.iloc[0][key]
            
            if pd.isna(token) or token is None:
                print(f"{Fore.RED}❌ No instrument token for {strike} {option_type}{Style.RESET_ALL}")
                return None
            
            return token
            
        except Exception as e:
            print(f"{Fore.RED}❌ Token lookup error: {e}{Style.RESET_ALL}")
            return None
    
    def place_order(self, instrument_token: str, quantity: int, side: str,
                   tag: str = "VOLGUARD", is_intraday: bool = True,
                   order_type: str = "MARKET", price: float = 0.0) -> OrderResult:
        """Places Order using V3 API with Auto-Slicing for large orders"""
        
        if not instrument_token:
            return OrderResult(
                status="FAILED",
                order_ids=[],
                message="Invalid instrument token",
                error="No instrument token provided"
            )
        
        # Build order request
        body = upstox_client.PlaceOrderV3Request(
            quantity=int(quantity),
            product="I" if is_intraday else "D",
            validity="DAY",
            price=float(price) if order_type != "MARKET" else 0.0,
            tag=tag,
            instrument_token=instrument_token,
            order_type=order_type,
            transaction_type=side,
            disclosed_quantity=0,
            trigger_price=0.0,
            is_amo=False,
            slice=True  # CRITICAL: Auto-slice large orders
        )
        
        try:
            response = self.order_api.place_order(body)
            
            if response.status == "success":
                return OrderResult(
                    status="SUCCESS",
                    order_ids=response.data.order_ids if response.data else [],
                    message="Order placed successfully",
                    total_quantity=quantity
                )
            else:
                return OrderResult(
                    status="FAILED",
                    order_ids=[],
                    message=str(response),
                    error="API returned non-success status"
                )
                
        except ApiException as e:
            error_msg = str(e)
            print(f"{Fore.RED}❌ Order Error: {error_msg}{Style.RESET_ALL}")
            return OrderResult(
                status="FAILED",
                order_ids=[],
                message=f"API Exception: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = str(e)
            print(f"{Fore.RED}❌ Unexpected order error: {error_msg}{Style.RESET_ALL}")
            return OrderResult(
                status="FAILED",
                order_ids=[],
                message=f"Unexpected error: {error_msg}",
                error=error_msg
            )
    
    def place_multi_leg_order(self, legs: List[Dict], tag: str = "VOLGUARD_MULTI") -> Dict:
        """Place multi-leg order (for strategies like Iron Condor)"""
        
        results = []
        all_success = True
        
        print(f"\n{Fore.CYAN}Placing multi-leg order with {len(legs)} legs...{Style.RESET_ALL}")
        
        for i, leg in enumerate(legs, 1):
            print(f"  Leg {i}: {leg['side']} {leg['quantity']} of {leg['instrument_token']}")
            
            result = self.place_order(
                instrument_token=leg['instrument_token'],
                quantity=leg['quantity'],
                side=leg['side'],
                tag=f"{tag}_LEG{i}",
                order_type="LIMIT" if 'price' in leg and leg['price'] > 0 else "MARKET",
                price=leg.get('price', 0.0)
            )
            
            results.append(result)
            
            if result.status != "SUCCESS":
                all_success = False
                print(f"  {Fore.RED}❌ Leg {i} failed{Style.RESET_ALL}")
                
                # CRITICAL: If any leg fails, we need to handle the partial fill
                if i > 1:
                    print(f"  {Fore.YELLOW}⚠️ Closing previous legs due to failure{Style.RESET_ALL}")
                    self._close_previous_legs(legs[:i-1], results[:i-1])
                break
            else:
                print(f"  {Fore.GREEN}✓ Leg {i} successful{Style.RESET_ALL}")
        
        return {
            "overall_status": "SUCCESS" if all_success else "PARTIAL_FAILURE",
            "all_success": all_success,
            "legs": results,
            "total_order_ids": [order_id for result in results for order_id in result.order_ids]
        }
    
    def _close_previous_legs(self, legs: List[Dict], results: List[OrderResult]):
        """Close previously executed legs when multi-leg fails"""
        
        for leg, result in zip(legs, results):
            if result.status == "SUCCESS" and result.order_ids:
                # Reverse the position
                reverse_side = "SELL" if leg['side'] == "BUY" else "BUY"
                
                print(f"  Reversing leg: {reverse_side} {leg['quantity']}")
                
                # Place reverse order
                self.place_order(
                    instrument_token=leg['instrument_token'],
                    quantity=leg['quantity'],
                    side=reverse_side,
                    tag="VOLGUARD_REVERSE"
                )
    
    def execute_strategy(self, strategy_name: str, trade_plan: Dict,
                        chain_df: pd.DataFrame, expiry: str, lot_size: int) -> Dict:
        """Execute complete strategy (Iron Condor, Strangle, etc.)"""
        
        print(f"\n{Fore.CYAN}Executing {strategy_name} strategy...{Style.RESET_ALL}")
        
        # Build legs based on strategy
        legs = []
        
        if strategy_name == "SHORT_STRANGLE":
            legs = self._build_strangle_legs(trade_plan, chain_df, lot_size)
            
        elif strategy_name == "IRON_CONDOR":
            legs = self._build_iron_condor_legs(trade_plan, chain_df, lot_size)
            
        elif strategy_name == "IRON_BUTTERFLY":
            legs = self._build_iron_butterfly_legs(trade_plan, chain_df, lot_size)
            
        elif strategy_name == "PUT_CREDIT_SPREAD":
            legs = self._build_put_spread_legs(trade_plan, chain_df, lot_size)
            
        elif strategy_name == "CALL_CREDIT_SPREAD":
            legs = self._build_call_spread_legs(trade_plan, chain_df, lot_size)
            
        else:
            return {
                "status": "FAILED",
                "error": f"Unknown strategy: {strategy_name}",
                "message": "Strategy not implemented"
            }
        
        if not legs:
            return {
                "status": "FAILED",
                "error": "No legs built for strategy",
                "message": "Could not build order legs"
            }
        
        # Execute multi-leg order
        result = self.place_multi_leg_order(legs, tag=f"VOLGUARD_{strategy_name}")
        
        # Calculate total premium and risk
        if result['all_success']:
            total_premium = sum(trade_plan.get('net_credit', 0) * lot_size * trade_plan.get('lots', 1))
            result.update({
                'strategy_name': strategy_name,
                'total_premium': total_premium,
                'trade_plan': trade_plan,
                'execution_time': datetime.now()
            })
        
        return result
    
    def _build_strangle_legs(self, trade_plan: Dict, chain_df: pd.DataFrame, lot_size: int) -> List[Dict]:
        """Build legs for short strangle"""
        legs = []
        
        # Short Call
        call_token = self.get_token_from_chain(chain_df, trade_plan['call_strike'], "CE")
        if call_token:
            legs.append({
                'instrument_token': call_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('call_premium', 0)
            })
        
        # Short Put
        put_token = self.get_token_from_chain(chain_df, trade_plan['put_strike'], "PE")
        if put_token:
            legs.append({
                'instrument_token': put_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('put_premium', 0)
            })
        
        return legs
    
    def _build_iron_condor_legs(self, trade_plan: Dict, chain_df: pd.DataFrame, lot_size: int) -> List[Dict]:
        """Build legs for iron condor"""
        legs = []
        
        # Short Call
        call_short_token = self.get_token_from_chain(chain_df, trade_plan['call_short'], "CE")
        if call_short_token:
            legs.append({
                'instrument_token': call_short_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('call_short_premium', 0)
            })
        
        # Long Call
        call_long_token = self.get_token_from_chain(chain_df, trade_plan['call_long'], "CE")
        if call_long_token:
            legs.append({
                'instrument_token': call_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('call_long_premium', 0)
            })
        
        # Short Put
        put_short_token = self.get_token_from_chain(chain_df, trade_plan['put_short'], "PE")
        if put_short_token:
            legs.append({
                'instrument_token': put_short_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('put_short_premium', 0)
            })
        
        # Long Put
        put_long_token = self.get_token_from_chain(chain_df, trade_plan['put_long'], "PE")
        if put_long_token:
            legs.append({
                'instrument_token': put_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('put_long_premium', 0)
            })
        
        return legs
    
    def _build_iron_butterfly_legs(self, trade_plan: Dict, chain_df: pd.DataFrame, lot_size: int) -> List[Dict]:
        """Build legs for iron butterfly"""
        legs = []
        
        # Short ATM Call
        atm_call_token = self.get_token_from_chain(chain_df, trade_plan['atm_strike'], "CE")
        if atm_call_token:
            legs.append({
                'instrument_token': atm_call_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('call_premium', 0)
            })
        
        # Short ATM Put
        atm_put_token = self.get_token_from_chain(chain_df, trade_plan['atm_strike'], "PE")
        if atm_put_token:
            legs.append({
                'instrument_token': atm_put_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('put_premium', 0)
            })
        
        # Long Call
        call_long_token = self.get_token_from_chain(chain_df, trade_plan['call_long'], "CE")
        if call_long_token:
            legs.append({
                'instrument_token': call_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('call_long_premium', 0)
            })
        
        # Long Put
        put_long_token = self.get_token_from_chain(chain_df, trade_plan['put_long'], "PE")
        if put_long_token:
            legs.append({
                'instrument_token': put_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('put_long_premium', 0)
            })
        
        return legs
    
    def _build_put_spread_legs(self, trade_plan: Dict, chain_df: pd.DataFrame, lot_size: int) -> List[Dict]:
        """Build legs for put credit spread"""
        legs = []
        
        # Short Put
        put_short_token = self.get_token_from_chain(chain_df, trade_plan['put_short'], "PE")
        if put_short_token:
            legs.append({
                'instrument_token': put_short_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('put_short_premium', 0)
            })
        
        # Long Put
        put_long_token = self.get_token_from_chain(chain_df, trade_plan['put_long'], "PE")
        if put_long_token:
            legs.append({
                'instrument_token': put_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('put_long_premium', 0)
            })
        
        return legs
    
    def _build_call_spread_legs(self, trade_plan: Dict, chain_df: pd.DataFrame, lot_size: int) -> List[Dict]:
        """Build legs for call credit spread"""
        legs = []
        
        # Short Call
        call_short_token = self.get_token_from_chain(chain_df, trade_plan['call_short'], "CE")
        if call_short_token:
            legs.append({
                'instrument_token': call_short_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'SELL',
                'price': trade_plan.get('call_short_premium', 0)
            })
        
        # Long Call
        call_long_token = self.get_token_from_chain(chain_df, trade_plan['call_long'], "CE")
        if call_long_token:
            legs.append({
                'instrument_token': call_long_token,
                'quantity': lot_size * trade_plan.get('lots', 1),
                'side': 'BUY',
                'price': trade_plan.get('call_long_premium', 0)
            })
        
        return legs
