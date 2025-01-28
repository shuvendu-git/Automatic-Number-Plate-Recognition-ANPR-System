import sqlite3

def create_dummy_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VehicleOwners (
            VehicleNumber TEXT PRIMARY KEY,
            OwnerName TEXT,
            RegistrationDate TEXT,
            MobileNumber TEXT
        )
    ''')

    # Insert dummy data
    dummy_data = [
        ("KA01AB1234", "Alice Smith", "2023-01-15", "9876543210"),
        ("TN02CD5678", "Bob Johnson", "2022-08-20", "8765432109"),
        ("AP03EF9012", "Charlie Brown", "2024-03-10", "7654321098"),
        ("KL04GH3456", "Diana Miller", "2021-11-05", "6543210987"),
        ("MH05IJ7890", "Ethan White", "2023-05-25", "5432109876"),
        ("TS06KL1234", "Sophia Taylor", "2022-12-01", "4321098765"),
        ("GJ07MN5678", "David Garcia", "2024-02-18", "3210987654"),
        ("HR08OP9012", "Olivia Rodriguez", "2021-07-08", "2109876543"),
        ("SNK8338A", "James Wilson", "2023-09-30", "1098765432"),
        ("UP10RS7890", "Emily Martinez", "2022-04-12", "9087654321"),
        ("MH12DE1433", "Liam Anderson", "2023-06-01", "8076543219"),
    ]

    cursor.executemany("INSERT OR IGNORE INTO VehicleOwners VALUES (?, ?, ?, ?)", dummy_data)
    conn.commit()
    conn.close()
    print("Dummy database created with table 'VehicleOwners'.")


if __name__ == '__main__':
    database_file = "vehicle_database.db"
    create_dummy_database(database_file)
