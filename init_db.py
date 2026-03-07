import sqlite3

def init_db():
    conn = sqlite3.connect('urban_safety.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS safety_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            near_miss_count INTEGER,
            risk_level TEXT,
            risk_score REAL,
            video_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()