# Recap des colonnes nécessaires :
## 1) Init_Win_bytes_forward
## 2) Total Length of Fwd Packets
## 3) Bwd Header Length
## 4) Destination Port
## 5) Subflow Fwd Bytes
## 6) Packet Length Std
## 7) Packet Length Variance
## 8) Bwd Packets/s
## 9) Average Packet Size
## 10) Bwd Packet Length Std


# https://connect.ed-diamond.com/GNU-Linux-Magazine/glmfhs-090/scapy-le-couteau-suisse-python-pour-le-reseau



from scapy.all import *
from scapy.layers.inet import TCP



##  filter='http', the name of the filtred protocol.
#   count=2 ,
#   iface='eth0',  the used interfaces : 'eth0', 'Wi-Fi'
#   timeout=10    # Time of capure in seconds

#############################################################
# step 1 : write the URL

# step 2 :  fin the IP adresss
import socket
IP_used = socket.gethostbyname('www.lemonde.fr')
print(" Found IP  = ", IP_used )

# step 3  : lunch sniff  - and make filtres :
#           a) protocol : http/https ou other
#           url/ip adress :

# Create the sniffer
# sniffer = sniff(timeout=10)
pkts_sniffed = sniff(filter='host' + IP_used, iface='Wi-Fi', timeout=10)

pkts_sniffed.summary()

wrpcap("MaCaptureMonde3.pcapng", pkts_sniffed)
print("Capture réussie")



'''
for pkt in pkts_sniffed:
    if TCP in pkt:
        print("* TRAME *")
        # comment les rÃ©cupÃ©rer un par un
        # print(pkt[TCP].sport)
        # print(pkt[TCP].dport)
        # afficher toute la trame
        print(pkt.display())

'''