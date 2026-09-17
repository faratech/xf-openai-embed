import os
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
db_config = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}

# SQL to create the openai_embeddings table
create_table_sql = """
CREATE TABLE IF NOT EXISTS openai_embeddings (
  id INT(11) NOT NULL AUTO_INCREMENT,
  post_id INT(10) UNSIGNED,
  thread_id INT(10) UNSIGNED,
  embedding BLOB,
  embedding_length INT(11),
  is_truncated TINYINT(1),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  section INT(11) DEFAULT 0,
  needs_update TINYINT(1) DEFAULT 0,
  PRIMARY KEY (id),
  KEY post_id (post_id),
  KEY thread_id (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

def setup_database():
    try:
        # Establish a database connection
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Create the table
        cursor.execute(create_table_sql)
        print("openai_embeddings table created successfully.")

        # Commit the changes
        conn.commit()

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    setup_database()
