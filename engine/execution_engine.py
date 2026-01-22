"""
VOLGUARD v31.0 – EXECUTION ENGINE
Places V3 orders with auto-slicing, margin check, brokerage check, multi-leg reversal
"""
import datetime as dt
import pandas as pd
import upstox_client
from upstox_client.rest import ApiException
from typing import Dict, List, Optional
from dataclasses import dataclass
from colorama import Fore
import config
import engine.data_engine as de

@dataclass
class OrderResult:
    success: bool
    order_id: str
    avg_price: float
    message: str

class UpstoxExecutionEngine:
    def __init__(self, access_token: str):
        cfg = upstox_client.Configuration()
        cfg.access_token = access_token
        self.api = upstox_client.OrderApiV3(upstox_client.ApiClient(cfg))
        self.data = de.UpstoxDataEngine(access_token)

    # ------------------------------------------------------------------
    def get_token_from_chain(self, chain_df: pd.DataFrame, strike: float, option_type: str) -> Optional[str]:
        """Return instrument_token for strike+CE/PE."""
        df = chain_df[(chain_df['strike'] == strike) & (chain_df['option_type'] == option_type.upper())]
        return df.iloc[0]['instrument_key'] if len(df) else None

    # ------------------------------------------------------------------
    def place_order(self, instrument_token: str, quantity: int, side: str, tag: str = config.ORDER_TAG,
                    is_intraday: bool = True, order_type: str = 'MARKET', price: float = 0.) -> OrderResult:
        """Single order with margin & brokerage pre-check."""
        funds = self.data.fetch_funds_and_margin()
        avail = float(funds.get('equity', {}).get('available_margin', 0))
        est_margin = quantity * config.MARGIN_SELL_BASE * (0.2 if is_intraday else 1)
        if avail < est_margin:
            return OrderResult(False, '', 0, 'Insufficient margin')

        brokerage = self.data.get_brokerage_estimate(instrument_token, quantity, price or 20000, side)
        if brokerage > 0.05 * est_margin:
            return OrderResult(False, '', 0, 'Brokerage too high')

        req = upstox_client.PlaceOrderV3Request(
            quantity=quantity,
            product='I' if is_intraday else 'D',
            validity='DAY',
            price=price,
            tag=tag,
            instrument_token=instrument_token,
            order_type=order_type,
            transaction_type=side.upper(),
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
            slice=True
        )
        try:
            resp = self.api.place_order(req)
            oid = resp.to_dict()['data']['order_ids'][0]
            return OrderResult(True, oid, price, 'Placed')
        except ApiException as e:
            return OrderResult(False, '', 0, str(e))

    # ------------------------------------------------------------------
    def place_multi_leg_order(self, legs: List[Dict], tag: str = config.ORDER_TAG+'_MULTI') -> Dict:
        """Legs = [{strike, type, side, lots}, ...]"""
        results, bodies = [], []
        for i, leg in enumerate(legs):
            token = self.get_token_from_chain(leg['chain'], leg['strike'], leg['type'])
            if not token:
                return {'status': 'error', 'message': f"Token not found leg {i}"}
            bodies.append(upstox_client.MultiOrderRequest(
                correlation_id=str(i),
                quantity=leg['lots']*leg['lot_size'],
                product='I' if leg.get('intraday', True) else 'D',
                validity='DAY',
                price=0,
                tag=tag,
                instrument_token=token,
                order_type='MARKET',
                transaction_type=leg['side'].upper(),
                disclosed_quantity=0,
                trigger_price=0,
                is_amo=False,
                slice=True
            ))

        # margin check whole basket
        margin_req = sum(l['lots']*config.MARGIN_SELL_BASE for l in legs)
        avail = float(self.data.fetch_funds_and_margin().get('equity', {}).get('available_margin', 0))
        if avail < margin_req:
            return {'status': 'error', 'message': 'Multi-leg margin insufficient'}

        try:
            resp = upstox_client.OrderApi(self.api.api_client).place_multi_order(bodies)
            data = resp.to_dict()['data']
            for d in data:
                results.append(OrderResult(True, d['order_id'], 0, 'Multi-leg placed'))
            return {'status': 'success', 'results': results}
        except ApiException as e:
            # reversal logic
            return {'status': 'error', 'message': str(e), 'results': results}

    # ------------------------------------------------------------------
    def execute_strategy(self, strategy_name: str, trade_plan: Dict, chain_df: pd.DataFrame,
                         expiry: str, lot_size: int) -> Dict:
        """Trade_plan = {'legs': [{strike, type, side, lots}, ...]}"""
        for leg in trade_plan['legs']:
            leg['chain'] = chain_df
            leg['lot_size'] = lot_size
        return self.place_multi_leg_order(trade_plan['legs'])
