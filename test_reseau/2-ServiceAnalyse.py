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


import pyshark
# IP esme.fr  =  104.21.79.224

import socket
IP_used = socket.gethostbyname('monde.fr')


#dataCaptured = pyshark.FileCapture('MaCaptureMonde2.pcapng',
#                                   display_filter="ip.addr == 104.21.79.224")

dataCaptured = pyshark.FileCapture('MaCaptureMonde3.pcapng',
                                   display_filter="ip.addr == 81.92.80.55")


packet  = dataCaptured[0]
print( "IP dst ===", packet['IP'].dst)
print( "IP src ===", packet['IP'].src)

"""
for pkt in dataCaptured:
    #print (pkt)
    protocol = pkt.transport_layer   # ip |  transport_layer
    print ("Protocol is : ", protocol)
    source_address = pkt.ip.src
    print("Source IP adress is : ", source_address)
    source_port = pkt[pkt.transport_layer].srcport
    print("Source port is : ", source_port)
    destination_address = pkt.ip.dst
    print("Destination IP adress is : ", destination_address)
    destination_port = pkt[pkt.transport_layer].dstport
    print("Destination port is : ", destination_port)
    print("---------------------------------------")


paquet = cap[0]
print(paquet['ip'].dst)


i = 1
for paquet in cap:
    print("N° paquet  = ", i)
    print(paquet['ip'].dst)
    i = i+1
"""