# utils/auth.py
import bcrypt
import secrets
import sqlite3

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_default_admin(conn):
    admin_pw = secrets.token_urlsafe(12)
    hashed = hash_password(admin_pw)
    conn.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                 ("admin", hashed, "admin"))
    conn.commit()
    print(f"Default admin password: {admin_pw} (change immediately!)")
