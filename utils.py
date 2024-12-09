# function to format large numbers
def format_large_number(num):
    if num >= 1_000_000_000_000:  # Trillions
        return f"{num / 1_000_000_000_000:.1f}T"
    elif num >= 1_000_000_000:  # Billions
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:  # Millions
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:  # Thousands
        return f"{num / 1_000:.1f}K"
    else:  # Less than 1,000
        return str(num)


def format_colored_number(value):
    color = "green" if value >= 0 else "red"  # Choose green for positive, red for negative
    return f'<div style="color:{color}; font-size:20px; font-weight:bold;">{value * 100:.1f}%</div>'  # Format number to 2 decimal places

def safe_format(value):
        # Handle invalid or missing values
        if isinstance(value, (int, float)):
            return f"{value * 100:.1f}%"
        else:
            return "N/A"
       
