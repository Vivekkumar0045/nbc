import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_bearing_data(num_units=200):
    data = []
    
    # Simulation parameters
    # Generates data with non-linear relationships and physics-based failure signatures
    
    for unit_id in range(1, num_units + 1):
        # 1. Determine Operational Context
        rpm = np.random.normal(1500, 300) # Varied operational speeds
        load_factor = np.random.uniform(0.5, 1.2) # Load on the machine (50% to 120%)
        
        # 2. Assign Health State (Lifecycle Stage)
        # 0.0 (New) -> 1.0 (Failed) - Beta distribution skews towards healthy
        degradation = np.random.beta(2, 5) 
        
        # 3. Define Failure Mode (affects sensor signature)
        failure_mode = np.random.choice(
            ['None', 'Inner Race Defect', 'Outer Race Defect', 'Ball Defect', 'Lubrication Failure'], 
            p=[0.5, 0.15, 0.15, 0.1, 0.1]
        )
        
        if degradation < 0.2: failure_mode = 'None' # Force healthy if low degradation
        
        # 4. Generate Sensor Data with Physics-based correlations
        
        # Base readings (Healthy State)
        base_vib = (rpm / 1000) * 0.5 * load_factor # Higher RPM/Load = Higher natural vibration
        base_temp = 40 + (rpm / 1000) * 5 + (load_factor * 10)
        
        # Apply degradation effects
        if failure_mode == 'None':
            vib = base_vib * (1 + degradation * 0.5) + np.random.normal(0, 0.05)
            temp = base_temp * (1 + degradation * 0.2) + np.random.normal(0, 1)
            lub = 100 - (degradation * 50) + np.random.normal(0, 5)
            
        elif failure_mode == 'Inner Race Defect':
            # High frequency vibration spikes
            vib = base_vib * (1 + degradation * 6) + np.random.normal(0, 0.2)
            temp = base_temp * (1 + degradation * 0.5)
            lub = 90 - (degradation * 30)
            
        elif failure_mode == 'Outer Race Defect':
            # Moderate vibration, audible noise
            vib = base_vib * (1 + degradation * 4) + np.random.normal(0, 0.15)
            temp = base_temp * (1 + degradation * 0.3)
            lub = 90 - (degradation * 30)
            
        elif failure_mode == 'Ball Defect':
            # Irregular vibration
            vib = base_vib * (1 + degradation * 5) + np.random.normal(0, 0.5) # Noisy
            temp = base_temp * (1 + degradation * 0.4)
            lub = 85 - (degradation * 20)
            
        elif failure_mode == 'Lubrication Failure':
            # High Temp, Friction
            lub = max(0, 50 - (degradation * 50) - np.random.uniform(0, 20))
            friction_heat = (100 - lub) * 0.5 * load_factor
            temp = base_temp + friction_heat
            vib = base_vib * (1 + degradation * 2) # Vibration starts later
            
        # 5. Calculate RUL (Non-linear inverse function of specific stress)
        # A "Stress Score" combines anomalies
        vib_score = max(0, vib - base_vib) / base_vib
        temp_score = max(0, temp - base_temp) / base_temp
        
        total_stress = (vib_score * 2) + temp_score # Vibration weighted higher
        
        # Non-linear RUL decay
        if total_stress < 0.2:
            rul = np.random.randint(300, 800) # Healthy
            status = 'Healthy'
        elif total_stress < 1.0:
            rul = int(300 * np.exp(-2 * total_stress)) + np.random.randint(0, 30)
            status = 'Degrading'
        else:
            rul = int(50 * np.exp(-1 * total_stress))
            status = 'Critical'
            
        # Refine RUL based on operating hours (older bearings fail faster)
        hours = np.random.randint(500, 25000)
        age_factor = 1 - (hours / 50000) # Very simple age penalty
        rul = int(rul * age_factor)
        
        record = {
            'Unit_ID': f"NBC-{unit_id:04d}",
            'Location': np.random.choice(['Plant A', 'Plant B', 'Wind Farm X', 'Solar Park Y']),
            'Operating_Hours': hours,
            'RPM': round(rpm, 0),
            'Load_Factor': round(load_factor, 2),
            'Temperature_C': round(temp, 1),
            'Vibration_mm_s': round(vib, 2),
            'Lubrication_Level_Pct': int(max(0, min(100, lub))),
            'Failure_Mode': failure_mode,
            'Status': status,
            'RUL_Days': max(0, rul) # TARGET VARIABLE
        }
        data.append(record)
        
    df = pd.DataFrame(data)
    file_path = 'bearings_data.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated {file_path} with {len(df)} records.")
    return df

def generate_dealer_data(num_dealers=20):
    dealers = []
    locations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Pune', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Jamshedpur']
    
    for dealer_id in range(1, num_dealers + 1):
        dealer = {
            'Dealer_ID': f"DLR-{dealer_id:03d}",
            'Name': f"Apex Industrial {dealer_id}",
            'Location': np.random.choice(locations),
            'Inventory_Level': np.random.randint(0, 500),
            'Service_Responsiveness_Score': round(np.random.uniform(3.0, 5.0), 1), # 1-5 Scale
            'Turnaround_Time_Hrs': np.random.randint(4, 48),
            'Technical_Capability_Score': round(np.random.uniform(60, 100), 1),
            'Customer_Satisfaction_Index': round(np.random.uniform(70, 100), 1)
        }
        dealers.append(dealer)
        
    df = pd.DataFrame(dealers)
    file_path = 'dealer_network.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated {file_path} with {len(df)} records.")
    return df

if __name__ == "__main__":
    generate_bearing_data(200)
    generate_dealer_data(20)
