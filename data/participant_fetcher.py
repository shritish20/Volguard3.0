"""
FII/DII/Pro/Client data fetcher from NSE
Your original code integrated
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import io
import pytz
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from analytics.models import ParticipantData


class ParticipantDataFetcher:
    """Fetch FII/DII/Pro/Client data from NSE archives"""
    
    @staticmethod
    def get_candidate_dates(days_back: int = 5):
        """Get candidate dates for data fetching"""
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        dates = []
        candidate = now
        if candidate.hour < 18:
            candidate -= timedelta(days=1)
        while len(dates) < days_back:
            if candidate.weekday() < 5:
                dates.append(candidate)
            candidate -= timedelta(days=1)
        return dates
    
    @staticmethod
    def fetch_oi_csv(date_obj):
        """Fetch OI CSV from NSE archives"""
        date_str = date_obj.strftime('%d%m%Y')
        url = f"https://archives.nseindia.com/content/nsccl/fao_participant_oi_{date_str}.csv"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
                "Connection": "keep-alive"
            }
            r = requests.get(url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                content = r.content.decode('utf-8')
                if "Future Index Long" in content:
                    lines = content.splitlines()
                    for idx, line in enumerate(lines[:20]):
                        if "Future Index Long" in line:
                            df = pd.read_csv(io.StringIO(content), skiprows=idx)
                            df.columns = df.columns.str.strip()
                            return df
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def process_participant_data(df) -> Dict[str, ParticipantData]:
        """Process participant data from DataFrame"""
        data = {}
        
        for participant in ["FII", "DII", "Client", "Pro"]:
            try:
                # Find row for this participant
                mask = df['Client Type'].astype(str).str.contains(
                    participant, case=False, na=False
                )
                row = df[mask].iloc[0]
                
                # Helper function to parse values
                def get_val(col):
                    val = str(row[col])
                    # Remove commas and convert to float
                    return float(val.replace(',', '')) if val else 0.0
                
                # Create ParticipantData object
                data[participant] = ParticipantData(
                    fut_long=get_val('Future Index Long'),
                    fut_short=get_val('Future Index Short'),
                    fut_net=get_val('Future Index Long') - get_val('Future Index Short'),
                    call_long=get_val('Option Index Call Long'),
                    call_short=get_val('Option Index Call Short'),
                    call_net=get_val('Option Index Call Long') - get_val('Option Index Call Short'),
                    put_long=get_val('Option Index Put Long'),
                    put_short=get_val('Option Index Put Short'),
                    put_net=get_val('Option Index Put Long') - get_val('Option Index Put Short'),
                    stock_net=get_val('Future Stock Long') - get_val('Future Stock Short')
                )
            except Exception:
                data[participant] = None
        
        return data
    
    @classmethod
    def fetch_smart_participant_data(cls) -> Tuple[Optional[Dict], Optional[Dict], float, str, bool]:
        """
        Fetch participant data with fallback logic
        
        Returns:
            Tuple of (primary_data, secondary_data, fii_net_change, data_date, is_fallback)
        """
        print("  📊 Connecting to NSE Archives...")
        dates = cls.get_candidate_dates()
        
        primary_data = None
        primary_date = None
        secondary_data = None
        
        for d in dates:
            print(f"     > Trying {d.strftime('%d-%b')}...", end=" ")
            df = cls.fetch_oi_csv(d)
            
            if df is not None:
                primary_data = cls.process_participant_data(df)
                primary_date = d
                print(f"FOUND")
                
                # Try to get previous day data for net change calculation
                prev = d - timedelta(days=1)
                while prev.weekday() >= 5:  # Skip weekends
                    prev -= timedelta(days=1)
                
                df_prev = cls.fetch_oi_csv(prev)
                if df_prev is not None:
                    secondary_data = cls.process_participant_data(df_prev)
                
                break
            else:
                print(f"MISSING")
        
        # Handle no data case
        if primary_data is None:
            return None, None, 0.0, "NO DATA", False
        
        # Calculate FII net change
        fii_net_change = 0.0
        if primary_data.get('FII') and secondary_data and secondary_data.get('FII'):
            fii_net_change = (
                primary_data['FII'].fut_net - secondary_data['FII'].fut_net
            )
        
        # Check if using fallback data
        is_fallback = primary_date.date() != dates[0].date()
        date_str = primary_date.strftime('%d-%b-%Y')
        
        return primary_data, secondary_data, fii_net_change, date_str, is_fallback
