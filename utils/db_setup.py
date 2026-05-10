import sqlite3
import os

def init_db():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, 'data', 'mro_assets.db')
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create MRO History table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mro_history (
        engine_id INTEGER PRIMARY KEY,
        last_overhaul_cycle INTEGER,
        last_maintenance_action TEXT,
        technician_notes TEXT
    )
    ''')
    
    # Sample data
    mro_data = [
        (1, 150, "HPC Blade Replacement", "Engine overhauled 150 cycles ago. All compressor blades checked."),
        (2, 210, "Fuel Metering Unit (FMU) Swap", "FMU replaced due to erratic phi readings. Calibrated for FD001 conditions."),
        (3, 50, "A-Check", "Standard pneumatic system flush and filter replacement."),
        (4, 300, "Full Overhaul", "Complete disassembly and inspection. Engine in pristine condition post-service."),
        (5, 120, "Fan Blade Inspection", "Borescope inspection showed minor pitting. Safe for continued operation.")
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO mro_history VALUES (?,?,?,?)', mro_data)
    
    conn.commit()
    conn.close()
    print(f"SQL Database initialized at: {db_path}")

if __name__ == "__main__":
    init_db()
