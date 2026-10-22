import pymysql
from xf_embed.config import settings

# Use the same loader and driver family as the search application.
db_config = {
    "host": settings.MYSQL_HOST,
    "port": settings.MYSQL_PORT,
    "user": settings.MYSQL_USER,
    "password": settings.MYSQL_PASSWORD,
    "database": settings.MYSQL_DATABASE,
    "charset": settings.MYSQL_CHARSET,
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
  thread_embedding_key INT(10) UNSIGNED GENERATED ALWAYS AS
    (CASE WHEN post_id IS NULL THEN thread_id ELSE NULL END) VIRTUAL,
  PRIMARY KEY (id),
  UNIQUE KEY post_id (post_id, section),
  UNIQUE KEY uq_thread_embedding_key (thread_embedding_key),
  KEY thread_id (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

def setup_database():
    # CREATE IF NOT EXISTS preserves existing tables; it never alters live keys.
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print("openai_embeddings table created successfully.")
    except pymysql.MySQLError as err:
        print(f"Error: {err}")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    setup_database()
