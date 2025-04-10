import subprocess

COMPONENT_COUNT = 12

def watt_now():
    result = subprocess.run(['sudo', 'vcgencmd', 'pmic_read_adc'], capture_output=True, text=True)
    data = result.stdout.split('\n')

    watt_sum = 0
    for i in range(COMPONENT_COUNT):
        amp = data[i].split('=')[1].replace('A', '')
        volt = data[i+COMPONENT_COUNT].split('=')[1].replace('V', '')
        watt_sum += float(amp)*float(volt)

    return watt_sum

if __name__ == '__main__':
    print(f'Currently using: {watt_now():.3f} watts')