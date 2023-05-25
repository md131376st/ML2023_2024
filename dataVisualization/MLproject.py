import numpy
import matplotlib.pyplot as plt

def readingdata(): 

    TrainData = []
    TrainLable = []
    with open('ZohCoding/Train.txt') as TraintFile:
        
        for line in TraintFile:
            D = line.split(",")
            del D[12]
            Data = [float(x) for x in D]
            TrainData.append(Data)
            
            l = line.split(",")[12].rsplit()
            for x in l:
                Lable = int(x) 
                TrainLable.append(Lable)

    TrainData_array = numpy.array(TrainData).T
    TrainLable_array = numpy.array(TrainLable)

    return (TrainData_array,TrainLable_array)
    # print(TrainLable_array)
    # print(TrainData_array.shape)
       
                
# import sklearn.datasets
# data , label=  sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
# # print((data,label))
# print(data)
# print(label)



def vrow(x):
    return x.reshape(1, x.size)


def vcol(x):
    return x.reshape(x.size, 1)

def emprialmean(D):
    return vcol(D.mean(1))


def Covariance(D):
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1))
    C = numpy.dot(DC, DC.T)/ DC.shape[1]
    return C

def PCA(D, m):
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1))
    C = numpy.dot(DC, DC.T)/ DC.shape[1]
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    return numpy.dot(P.T, D)



def plot_scatter(D, L):
    Men = D[:, L==0]
    women = D[:, L==1]
    # D2 = D[:, L==2]

    plt.figure()
    plt.scatter(Men[0, :], Men[1, :], label = 'men')
    plt.scatter(women[0, :], women[1, :], label = 'women')
    # plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
        
    plt.show()


# D, L = readingdata()
# DP = PCA(D, 2)

# plot_scatter(DP, L)


def LoadData():
        data = numpy.genfromtxt("Train.txt", delimiter=",")
        return (data[:, -1].T, data[:, :-1].T)

L, D = LoadData()
def logpdf_GAU_ND(X, mu, C):
    p = numpy.linalg.inv(C)
    return -0.5*X.shape[0]*numpy.log(numpy.pi*2) + 0.5* numpy.linalg.slogdet(p)[1]- 0.5*(numpy.dot(p,(X-mu))*(X-mu)).sum(0)


def ML_GAU(D):
    mu = vcol(D.mean(1))
    C = numpy(D-mu,(D-mu).T)/float(D.shape[1])
    return (mu, C)


 

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

print(vrow(D).shape)
plt.figure()
XPlot = numpy.linspace(-8, 12, 1000)
mu = emprialmean(D)
C = Covariance(D)
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
plt.show()
plot_scatter(L,D)