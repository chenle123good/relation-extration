import matplotlib
import matplotlib.pyplot as plt
import decimal

decimal.getcontext().prec = 3
model_name=['PCNN_ONE','PCNN_ATT']

for i in range(1,16):
    pcnn_one_p,pcnn_one_r,pcnn_att_p,pcnn_att_r=[],[],[],[]
    out1=open('./out/{}_{}_{}_PR.txt'.format(model_name[0], 'DEF', i),'r')
    out2 = open('./out/{}_{}_{}_PR.txt'.format(model_name[1], 'DEF', i), 'r')
    for line1 in out1.readlines():
        if line1!='\n':
            line1=line1.split()
            pcnn_one_p.append(decimal.Decimal(line1[0]))
            pcnn_one_r.append(decimal.Decimal(line1[1]))
    for line2 in out2.readlines():
        if line2!='\n':
            line2 = line2.split()
            pcnn_att_p.append(decimal.Decimal(line2[0]))
            pcnn_att_r.append(decimal.Decimal(line2[1]))
out1.close()
out2.close()

print('开始绘图---------------------')
plt.figure()
plt.xlabel("recall")
plt.ylabel("precision")
plt.plot( pcnn_one_r,pcnn_one_p)
plt.plot(pcnn_att_r, pcnn_att_p)
plt.legend(labels=['PCNN_ONE_Large','PCNN_ATT_Large'],loc="best")
plt.savefig('PCNN_ONE_ATT_large.png')
plt.show()


