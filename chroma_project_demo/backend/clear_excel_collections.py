#!/usr/bin/env python3
"""
Quick utility to delete all ChromaDB collections related to Excel uploads.

Run this script from the `backend` directory after changing your embedding model
to clear out old vector data with incompatible dimensions.

Usage:
  python clear_excel_collections.py
"""

import os
import sys

# Ensure the script can find the 'database' module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_chroma_client

def clear_excel_collections():
    print("Connecting to ChromaDB...")
    client = get_chroma_client()
    if not client:
        print("Error: Could not connect to ChromaDB. Please check your configuration.")
        return

    print("Fetching all collections...")
    collections = client.list_collections()
    excel_collections = [c for c in collections if getattr(c, 'name', '').startswith('excel_data_')]

    if not excel_collections:
        print("No Excel-related collections ('excel_data_*') found to delete.")
        return

    print(f"Found {len(excel_collections)} Excel collections to delete:")
    for c in excel_collections:
        print(f"  - {c.name}")

    confirm = input("Are you sure you want to delete these collections? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    for c in excel_collections:
        try:
            print(f"Deleting collection '{c.name}'...")
            client.delete_collection(name=c.name)
            print(f"Successfully deleted '{c.name}'.")
        except Exception as e:
            print(f"Error deleting collection '{c.name}': {e}")

    print("\nCleanup complete.")

if __name__ == "__main__":
    clear_excel_collections()

