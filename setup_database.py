#!/usr/bin/env python3
"""
Script to set up the database search functions for the Mosaic API.
This creates the missing vector_search and bm25_search RPC functions.
"""

import os
from supabase import create_client

def setup_database():
    """Set up the database with required search functions."""
    
    # Get environment variables
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_service_key:
        print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables")
        return False
    
    # Create Supabase client
    supabase = create_client(supabase_url, supabase_service_key)
    
    # Read the SQL functions file
    with open('db/search_functions.sql', 'r') as f:
        sql_content = f.read()
    
    try:
        # Execute the SQL to create functions
        result = supabase.rpc('exec_sql', {'sql': sql_content})
        print("Successfully created database search functions")
        return True
    except Exception as e:
        print(f"Error creating database functions: {e}")
        
        # Try alternative approach - execute each function separately
        functions = sql_content.split('-- ')
        for i, func in enumerate(functions):
            if func.strip():
                try:
                    # Use raw SQL execution
                    print(f"Executing function {i+1}...")
                    # Note: This approach may not work with Supabase client
                    # We may need to use the Supabase dashboard SQL editor instead
                    print(f"SQL to execute:\n{func}")
                except Exception as func_error:
                    print(f"Error executing function {i+1}: {func_error}")
        
        return False

if __name__ == "__main__":
    success = setup_database()
    if success:
        print("Database setup completed successfully!")
    else:
        print("Database setup failed. Please run the SQL manually in Supabase dashboard.")
        print("\nSQL to execute:")
        with open('db/search_functions.sql', 'r') as f:
            print(f.read())
