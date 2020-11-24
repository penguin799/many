import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm

# クラスタ数
N_CLUSTERS = 10

f = open('data.txt', 'r')
lines = f.read().split()
f.close()
dataset=[]
x=[]
y=[]

#strからfloatへ変換
for i in range(int(len(lines)/2)):
    x.append(float(lines[2*i]))
    y.append(float(lines[2*i+1]))
    dataset.append([float(lines[2*i]),float(lines[2*i+1])])

plt.scatter(x,y)
plt.show()

#適当に選ぶ
NumList=list(range(len(dataset)))
random.shuffle(NumList)

#N分割用
centers=[]
for i in range(N_CLUSTERS):
    centers.append(dataset[NumList.pop()])

#行列に変換
centers=np.array(centers)
dataset=np.array(dataset)

#共分散行列の計算
CovData=np.cov(dataset,rowvar=0,bias=0)
CovInv=np.linalg.inv(CovData)
while 1:
    #属する点を保存するリスト
    S=[[] for i in range(N_CLUSTERS)]

    #マハラノビス距離の計算とSへの格納
    for i in range(len(dataset)):
        Mahara=[]
        for j in range(len(centers)):
            Maharanobis=np.dot(np.dot((dataset[i]-centers[j]).T,CovInv),(dataset[i]-centers[j]))
            Mahara.append(Maharanobis)

        num=Mahara.index(min(Mahara))
        S[num].append(dataset[i])

    S=np.array(S)
    #色を付けてプロット
    for i in range(N_CLUSTERS):
        for j in range(len(S[i])):
            plt.scatter(S[i][j][0],S[i][j][1],color=cm.hsv(float(i)/N_CLUSTERS))

    # 代表パタンプロット
    plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')

    plt.pause(0.5)
    plt.clf()

    #セントロイドを求める
    end = 1
    for i in range(N_CLUSTERS):
        x=0
        y=0
    #重心を求める
        for j in range(len(S[i])):
            x=x+S[i][j][0]
            y=y+S[i][j][1]

        X=x/len(S[i])
        Y=y/len(S[i])
        mean=[X,Y]
        mean = np.array(mean, dtype='float32')

        first = 1
        for j in range(len(S[i])):
            Maharanobis = np.dot(np.dot(S[i][j] - mean.T, CovInv), (S[i][j] - mean))
            if first==1:
                first=0
                mahara=Maharanobis
                newcenters=S[i][j]

            if mahara>Maharanobis:
                mahara=Maharanobis
                newcenters=S[i][j]

        #セントロイドを更新
        if not(all(centers[i]==newcenters)):
            centers[i] = newcenters
            end=0

    if (end == 1):#終了時に結果を出力
        for i in range(N_CLUSTERS):
            for j in range(len(S[i])):
                plt.scatter(S[i][j][0], S[i][j][1], color=cm.hsv(float(i) / N_CLUSTERS))

        # 代表パタンプロット
        plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')

        print("Finish")
        plt.show()
        break

    #色を付けてプロット
    for i in range(N_CLUSTERS):
        for j in range(len(S[i])):
            plt.scatter(S[i][j][0],S[i][j][1],color=cm.hsv(float(i)/N_CLUSTERS))

    # 代表パタンプロット
    plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')

    plt.pause(0.5)
    plt.clf()   #初期化