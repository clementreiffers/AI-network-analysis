import pyshark

capture = pyshark.LiveCapture(interface='Wi-Fi', output_file='cpature_test_4.cap')
capture.sniff(timeout=10)
print(capture)

def print_result_cap():
    cap = pyshark.FileCapture('cpature_test.cap')
    print(cap[0])

# Capturer des packet tous les jours
# https://medium.com/@mwester1111/capturing-packets-continuously-everyday-using-pyshark-and-cron-a7835bf1beb0

