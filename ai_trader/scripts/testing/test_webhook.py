#!/usr/bin/env python3
"""
Test script for webhook alert functionality.
Usage: python test_webhook.py --webhook-url YOUR_WEBHOOK_URL
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main.monitoring.alerts.unified_alerts import UnifiedAlertSystem, AlertLevel, AlertCategory

async def test_webhook_alerts(webhook_url: str):
    """Test webhook alert sending."""
    # Create alert system with webhook configuration
    config = {
        'webhook': {
            'enabled': True,
            'url': webhook_url,
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    }
    
    alert_system = UnifiedAlertSystem(config)
    
    print(f"Testing webhook alerts to: {webhook_url}")
    print("-" * 50)
    
    # Test different alert types
    test_alerts = [
        {
            'title': 'Test Info Alert',
            'message': 'This is a test informational alert from AI Trader',
            'level': AlertLevel.INFO,
            'category': AlertCategory.SYSTEM
        },
        {
            'title': 'Test Trading Alert',
            'message': 'Test trade executed: BUY 100 AAPL at $150.00',
            'level': AlertLevel.INFO,
            'category': AlertCategory.TRADING,
            'metadata': {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.00}
        },
        {
            'title': 'Test Warning Alert',
            'message': 'Risk limit approaching: Portfolio VaR at 90% of limit',
            'level': AlertLevel.WARNING,
            'category': AlertCategory.RISK
        },
        {
            'title': 'Test Error Alert',
            'message': 'Failed to fetch market data for SPY',
            'level': AlertLevel.ERROR,
            'category': AlertCategory.DATA
        }
    ]
    
    # Send test alerts
    for i, alert_data in enumerate(test_alerts, 1):
        print(f"\n{i}. Sending {alert_data['level'].value} alert: {alert_data['title']}")
        
        alert_id = await alert_system.send_alert(
            title=alert_data['title'],
            message=alert_data['message'],
            level=alert_data['level'],
            category=alert_data['category'],
            metadata=alert_data.get('metadata', {})
        )
        
        if alert_id:
            print(f"   ✓ Alert sent successfully (ID: {alert_id})")
        else:
            print(f"   ✗ Failed to send alert")
        
        # Small delay between alerts
        await asyncio.sleep(1)
    
    # Test convenience methods
    print("\n5. Testing convenience method: alert_trade_executed")
    await alert_system.alert_trade_executed(
        symbol='MSFT',
        side='SELL',
        quantity=50,
        price=350.25,
        order_id='TEST-001'
    )
    print("   ✓ Trade alert sent")
    
    # Show alert statistics
    print("\n" + "-" * 50)
    print("Alert Statistics:")
    stats = alert_system.get_alert_stats()
    print(f"  Total alerts sent: {stats['total_alerts']}")
    print(f"  By category: {stats['by_category']}")
    print(f"  By level: {stats['by_level']}")
    
    # Cleanup
    await alert_system.cleanup()
    print("\n✓ Test completed and resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description='Test webhook alert functionality')
    parser.add_argument('--webhook-url', required=True, help='Webhook URL (e.g., Slack, Discord, Teams)')
    
    args = parser.parse_args()
    
    # Run test
    asyncio.run(test_webhook_alerts(args.webhook_url))

if __name__ == '__main__':
    main()