"""
Verify explosion detection is integrated in all signal files
"""
from pathlib import Path

def check_explosion_integration(file_path):
    """Check if file has explosion detection integrated"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key explosion detection components
        has_obv = 'def calculate_obv' in content
        has_money_flow = 'def money_flow_strength' in content
        has_cycle = 'def detect_memory_cycle_phase' in content
        has_acceleration = 'def trend_acceleration' in content
        has_filter = 'def explosive_trend_filter' in content
        has_sma_200 = "df['sma_200']" in content and "rolling(200).mean()" in content

        all_present = all([has_obv, has_money_flow, has_cycle, has_acceleration, has_filter, has_sma_200])

        return {
            'all_present': all_present,
            'obv': has_obv,
            'money_flow': has_money_flow,
            'cycle': has_cycle,
            'acceleration': has_acceleration,
            'filter': has_filter,
            'sma_200': has_sma_200
        }
    except Exception as e:
        return None

def main():
    """Main function"""
    print("=" * 80)
    print("Verifying Explosion Detection Integration")
    print("=" * 80)

    # Find all signal files
    current_dir = Path('.')
    files = sorted([f for f in current_dir.glob('get_trading_signal_*.py')
                   if '_bak' not in f.name])

    print(f"Checking {len(files)} files...\n")

    complete = 0
    incomplete = 0
    errors = 0
    missing_components = []

    for file_path in files:
        result = check_explosion_integration(file_path)

        if result is None:
            errors += 1
            print(f"[ERROR] {file_path.name}")
        elif result['all_present']:
            complete += 1
        else:
            incomplete += 1
            missing = [k for k, v in result.items() if k != 'all_present' and not v]
            missing_components.append((file_path.name, missing))

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:                   {len(files)}")
    print(f"Complete integration:          {complete}")
    print(f"Incomplete:                    {incomplete}")
    print(f"Errors:                        {errors}")
    print("=" * 80)

    if incomplete > 0:
        print("\nFiles with incomplete integration:")
        for filename, missing in missing_components:
            print(f"  {filename}: Missing {', '.join(missing)}")

    if complete == len(files):
        print(f"\n✅ All {len(files)} files have complete explosion detection integration!")
    else:
        print(f"\n⚠️  {incomplete} files need attention")

if __name__ == "__main__":
    main()
