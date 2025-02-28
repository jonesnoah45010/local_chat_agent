import sqlite3
from pathlib import Path

class local_sql_db:
    def __init__(self, db_path="local_database.db"):
        if len(db_path) <= 2 or ".db" != db_path[-2:]:
            db_path = db_path+".db"
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        print(self.db_path)
        self.connection = None
        self.connect()

    def connect(self):
        """Connect to the SQLite database."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        print(f"Connected to database at {self.db_path}")

    def create_table(self, table_name, schema):
        """
        Create a table in the database if it doesn't already exist.
        Args:
            table_name (str): Name of the table.
            schema (dict): Dictionary of column names and data types.
        Returns: 
            False if table was not created
            True if table was created
        Example:
                db_manager.create_table(
                    "users",
                    {
                        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                        "name": "TEXT NOT NULL",
                        "email": "TEXT UNIQUE NOT NULL",
                        "created_at": "DATETIME DEFAULT CURRENT_TIMESTAMP"
                    }
                )
        """
        # Check if the table already exists
        if self.table_exists(table_name):
            print(f"Table '{table_name}' already exists. Skipping creation.")
            return False

        # Create the table
        columns = ", ".join(f"{col} {dtype}" for col, dtype in schema.items())
        query = f"CREATE TABLE {table_name} ({columns});"
        self.run_query(query)
        print(f"Table '{table_name}' created.")
        return True

    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        Args:
            table_name (str): Name of the table to check.
        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = """
        SELECT name FROM sqlite_master WHERE type='table' AND name=?;
        """
        result = self.query_data(query, (table_name,))
        return len(result) > 0


    def get_schema(self, table_name):
        """
        Get the schema of a table in the format expected by create_table.
        Args:
            table_name (str): Name of the table.
        Returns:
            dict: Dictionary with column names as keys and data types as values.
        Example:
            schema = db_manager.get_schema("users")
            print(schema)
            # Output: {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "name": "TEXT NOT NULL", ...}
        """
        query = f"PRAGMA table_info({table_name});"
        result = self.query_data(query)

        if not result:
            print(f"Table '{table_name}' does not exist.")
            return None

        schema = {}
        for column in result:
            col_name = column[1]  # Column name
            col_type = column[2]  # Data type
            notnull = "NOT NULL" if column[3] else ""
            default = f"DEFAULT {column[4]}" if column[4] is not None else ""
            pk = "PRIMARY KEY AUTOINCREMENT" if column[5] else ""

            # Combine column attributes
            attributes = " ".join(filter(None, [col_type, notnull, default, pk]))
            schema[col_name] = attributes

        return schema




    def insert_data(self, table_name, data):
        """
        Insert data into a table.
        Args:
            table_name (str): Name of the table.
            data (list[dict]): List of dictionaries representing rows to insert.
        """
        if not data:
            print("No data provided for insert")
            return
        
        columns = ", ".join(data[0].keys())
        placeholders = ", ".join("?" for _ in data[0])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        values = [tuple(row.values()) for row in data]
        self.run_query(query, many=True, params=values)
        print(f"{len(data)} rows inserted into '{table_name}'.")

    def query_data(self, query, params=None, return_dict_list=True):
        """
        Query data from the database.
        Args:
            query (str): The SELECT query.
            params (tuple): Parameters for the query.
            return_dict_list (bool): Convert return value to list of dicts or use standard list of tuples
        Returns:
            list: Query results as a list of dicts, or list of tuples if return_dict_list=False.
        Example: 
            results = db_manager.query_data("SELECT * FROM users where name = ?",("James",))
        """
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())

        results = None
        if return_dict_list:
            column_names = [description[0] for description in cursor.description]
            results = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        else:
            results = cursor.fetchall()
        print(f"Query executed: {query}")
        return results

    def run_query(self, query, params=None, many=False):
        """
        Execute a SQL query.
        Example:
            insert_query = "INSERT INTO users (id, name, email) VALUES (?, ?, ?)"
            params = [
                (1, "Alice", "alice@fake.com"),
                (2, "Bob", "bob@fake.com"),
                (3, "Charlie", "charlie@fake.com")
            ]
            db.query(insert_query, params=params, many=True)
        """
        cursor = self.connection.cursor()
        if many:
            cursor.executemany(query, params)
        else:
            cursor.execute(query, params or ())
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print(f"Connection to {self.db_path} closed.")












# Example Usage
if __name__ == "__main__":
    db_manager = local_sql_db("example.db")
    
    # Create a table
    db_manager.create_table(
        "users",
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "name": "TEXT NOT NULL",
            "email": "TEXT UNIQUE NOT NULL",
            "created_at": "DATETIME DEFAULT CURRENT_TIMESTAMP"
        }
    )
    
    # Insert some data
    db_manager.insert_data(
        "users",
        [
            {"name": "James", "email": "james@example.com"},
            {"name": "Sam", "email": "sam@example.com"}
        ]
    )
    
    # Query the data
    results = db_manager.query_data("SELECT * FROM users where name = ?",("James",))
    for row in results:
        print(row)
    
    # Close the database connection
    db_manager.close()




















    
