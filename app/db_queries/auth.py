from supabase import create_client, Client
import os


def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client using environment variables.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "Supabase credentials not set. "
            "Please set SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    return create_client(url, key)


def signup_user(
    email: str,
    password: str,
    full_name: str,
    company_name: str,
    role: str,
    phone: str
) -> dict:
    """
    Registers a new user via Supabase Auth.

    Extra profile fields (full_name, company_name, role, phone) are stored
    in Supabase's user_metadata — no separate table needed.

    Args:
        email        : User's email address.
        password     : Plain-text password (Supabase hashes it).
        full_name    : User's full name.
        company_name : Company the user belongs to.
        role         : User's role e.g. 'sales_person', 'manager'.
        phone        : User's phone number.

    Returns:
        dict with user's uuid, email, and metadata.

    Raises:
        ValueError : If the email is already registered or input is invalid.
        RuntimeError: For unexpected Supabase errors.
    """
    supabase = get_supabase_client()

    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name"    : full_name,
                    "company_name" : company_name,
                    "role"         : role,
                    "phone"        : phone,
                }
            }
        })

        user = response.user

        if not user:
            raise ValueError("Signup failed — no user returned from Supabase.")

        return {
            "uuid"         : str(user.id),
            "email"        : user.email,
            "full_name"    : user.user_metadata.get("full_name"),
            "company_name" : user.user_metadata.get("company_name"),
            "role"         : user.user_metadata.get("role"),
            "phone"        : user.user_metadata.get("phone"),
            "created_at"   : str(user.created_at),
        }

    except Exception as e:
        error_msg = str(e)
        # Supabase returns this when email already exists
        if "already registered" in error_msg or "User already registered" in error_msg:
            raise ValueError("This email is already registered.")
        raise RuntimeError(f"Signup error: {error_msg}")


def login_user(email: str, password: str) -> dict:
    """
    Authenticates a user via Supabase Auth and returns JWT tokens + user info.

    Args:
        email    : Registered email address.
        password : Plain-text password.

    Returns:
        dict containing:
            - access_token  : JWT — frontend stores this and sends in Authorization header
            - token_type    : "bearer"
            - uuid          : User's unique ID
            - email         : User's email
            - full_name     : From user_metadata
            - company_name  : From user_metadata
            - role          : From user_metadata
            - phone         : From user_metadata
            - last_sign_in  : Timestamp of this login

    Raises:
        ValueError  : If credentials are wrong.
        RuntimeError: For unexpected Supabase errors.
    """
    supabase = get_supabase_client()

    try:
        response = supabase.auth.sign_in_with_password({
            "email"   : email,
            "password": password
        })

        user    = response.user
        session = response.session

        if not user or not session:
            raise ValueError("Invalid email or password.")

        return {
            "access_token" : session.access_token,
            "token_type"   : "bearer",
            "uuid"         : str(user.id),
            "email"        : user.email,
            "full_name"    : user.user_metadata.get("full_name"),
            "company_name" : user.user_metadata.get("company_name"),
            "role"         : user.user_metadata.get("role"),
            "phone"        : user.user_metadata.get("phone"),
            "last_sign_in" : str(user.last_sign_in_at),
        }

    except Exception as e:
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            raise ValueError("Invalid email or password.")
        raise RuntimeError(f"Login error: {error_msg}")