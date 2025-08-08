"""
Database Field Mappings

This module provides mappings between code field names and database column names
to handle cases where the existing database schema uses different names than the code expects.
"""

# Field mappings for the Company model
COMPANY_FIELD_MAPPINGS = {
    # Map code field name to database column name
    'avg_price': 'current_price',  # Code expects avg_price but DB has current_price
}

def map_company_fields(data: dict, to_db: bool = True) -> dict:
    """
    Map company fields between code and database representations.
    
    Args:
        data: Dictionary of field values
        to_db: If True, map from code to DB. If False, map from DB to code.
        
    Returns:
        Dictionary with mapped field names
    """
    mapped_data = data.copy()
    
    if to_db:
        # Mapping from code to database
        for code_field, db_field in COMPANY_FIELD_MAPPINGS.items():
            if code_field in mapped_data:
                mapped_data[db_field] = mapped_data.pop(code_field)
    else:
        # Mapping from database to code
        for code_field, db_field in COMPANY_FIELD_MAPPINGS.items():
            if db_field in mapped_data:
                mapped_data[code_field] = mapped_data.pop(db_field)
    
    return mapped_data