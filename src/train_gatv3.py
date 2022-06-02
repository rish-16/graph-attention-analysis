import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def LeakyRelu(x, beta):
    x[x < 0] *= beta 
    return x

def gamma_mlp_gat_fn(x_i, x_j, row1, row2, row3, row4):
    val1 = row1[0]*x_i + row1[1]*x_j
    val2 = row2[0]*x_i + row2[1]*x_j
    val3 = row3[0]*x_i + row3[1]*x_j
    val4 = row4[0]*x_i + row4[1]*x_j
    h    = np.asarray([1, 1, -1, -1])
    
    temp = np.asarray([val1,val2,val3,val4])
    temp = LeakyRelu(temp, 0.2)
    classification_of_pairs = h@temp
    temp = classification_of_pairs - max(classification_of_pairs)
    ex = np.exp(temp)
    
    return ex / np.sum(ex), classification_of_pairs

def gamma_gat_fn(x_i, x_j, a1, a2, b):
    temp = LeakyRelu(a1*x_i + a2*x_j + b, 0.2)
    temp = temp - max(temp)
    ex = np.exp(temp)
    return ex / np.sum(ex)

def all_gammas(X, adjlist, row1, row2, row3, row4, a1, a2, b):
    inter_gammas_mlp_gat = []
    intra_gammas_mlp_gat  = []
    inter_gammas_gat = [[[],[]],[[],[]]]
    intra_gammas_gat  = [[[],[]],[[],[]]]

    pairs_intra = 0
    pairs_inter = 0

    n_intra = 0
    n_inter = 0

    N = int(len(X)/2)
    x_mlp_gat = np.zeros(len(X))
    x_gat = np.zeros(len(X))
    x_gcn = np.zeros(len(X))
    
    for i in range(len(X)):
        
        gamma_gat_head1 = gamma_gat_fn(X[i], X[adjlist[i]], a1, a2, b)
        gamma_gat_head2 = gamma_gat_fn(X[i], X[adjlist[i]], -a1, -a2, -b)
        gamma_mlp_gat, pairs_mlp_gat = gamma_mlp_gat_fn(X[i], X[adjlist[i]], row1, row2, row3, row4)
        gamma_gcn = np.ones(len(adjlist[i]))/len(adjlist[i])
        
        x_mlp_gat[i] = np.dot(gamma_mlp_gat, X[adjlist[i]])
        x_gat[i] = np.dot(gamma_gat_head1, X[adjlist[i]]) + np.dot(gamma_gat_head2, X[adjlist[i]])
        x_gcn[i] = np.dot(gamma_gcn, X[adjlist[i]])
        
        ct = 0
        for j in adjlist[i]:
            if (j < N and i < N) or (j >= N and i >= N):
                intra_gammas_mlp_gat.append(gamma_mlp_gat[ct])
                pairs_intra += pairs_mlp_gat[ct] > 0
                
                if (j < N and i < N):
                    intra_gammas_gat[0][0].append(gamma_gat_head1[ct])
                    intra_gammas_gat[1][0].append(gamma_gat_head2[ct])
                else:
                    intra_gammas_gat[0][1].append(gamma_gat_head1[ct])
                    intra_gammas_gat[1][1].append(gamma_gat_head2[ct])
                n_intra += 1
            elif (j < N and i >= N) or (j >= N and i < N):
                inter_gammas_mlp_gat.append(gamma_mlp_gat[ct])
                pairs_inter += pairs_mlp_gat[ct] <= 0
                if (j < N and i >= N):
                    inter_gammas_gat[0][0].append(gamma_gat_head1[ct])
                    inter_gammas_gat[1][0].append(gamma_gat_head2[ct])
                else:
                    inter_gammas_gat[0][1].append(gamma_gat_head1[ct])
                    inter_gammas_gat[1][1].append(gamma_gat_head2[ct])                   
                n_inter += 1
            ct += 1
            
    class_pair_intra = pairs_intra/n_intra
    class_pair_inter = pairs_inter/n_inter
            
    return x_mlp_gat, x_gat, x_gcn, class_pair_intra, class_pair_inter, intra_gammas_mlp_gat, inter_gammas_mlp_gat, intra_gammas_gat, inter_gammas_gat

n = 1000
d = int(np.ceil(n/(np.log(n)**2)))
p = 0.5
q = 0.1
heads = 2 # does not work for another number

sizes = [int(n/2), int(n/2)]
probs = [[p,q], [q,p]]

std_ = 0.1
mu_up = 20*std_*np.sqrt(np.log(n**2))/(2*np.sqrt(d))
mu_lb = 0.01*std_/(2*np.sqrt(d))

mus = np.geomspace(mu_lb, mu_up, 30, endpoint=True)

ground_truth = np.concatenate((np.zeros(int(n/2)), np.ones(int(n/2))))

intra_avg_mlp_gat = []
intra_std_mlp_gat = []
inter_avg_mlp_gat = [] 
inter_std_mlp_gat = [] 

intra_avg_gat = [[[],[]],[[],[]]]
intra_std_gat = [[[],[]],[[],[]]]
inter_avg_gat = [[[],[]],[[],[]]]
inter_std_gat = [[[],[]],[[],[]]]

acc_mlp_gat = []
acc_gat = []
acc_gcn = []
acc_lin = []

acc_mlp_gat_std = []
acc_gat_std = []
acc_gcn_std = []
acc_lin_std = []

acc_pairs_intra = []
acc_pairs_inter = []

acc_pairs_intra_std = []
acc_pairs_inter_std = []

trials = 10

for mu in mus:
    
    sum_intra_avg_mlp_gat = 0
    sum_intra_std_mlp_gat = 0
    sum_inter_avg_mlp_gat = 0
    sum_inter_std_mlp_gat = 0
    
    sum_intra_avg_gat = [[0,0],[0,0]]
    sum_intra_std_gat = [[0,0],[0,0]]
    sum_inter_avg_gat = [[0,0],[0,0]]
    sum_inter_std_gat = [[0,0],[0,0]]
    
    pairs_intra_list = []
    pairs_inter_list = []
    
    acc_mlp_gat_list = []
    acc_gat_list = []
    acc_gcn_list = []
    acc_lin_list = []
    
    for trial in range(trials):
        g = nx.stochastic_block_model(sizes, probs)

        adjlist = [[v for v in g.neighbors(i)] for i in range(n)]
        for i in range(len(adjlist)):
            adjlist[i].append(i)

        X = np.zeros((n,d))
        X[:int(n/2)] = -mu
        X[int(n/2):] = mu
        noise = std_*np.random.randn(n,d)
        X = X + noise

        R = 1
        mu_ = mu*np.ones(d)
        w = (R/np.linalg.norm(mu_))*mu_
        Xw = X@w

        wmu_ = 1
        row1 = np.asarray([wmu_,wmu_])
        row2 = np.asarray([-wmu_,-wmu_])
        row3 = np.asarray([wmu_,-wmu_])
        row4 = np.asarray([-wmu_,wmu_])
        
        a1 = R/np.sqrt(2.0)
        a2 = a1
        b = -np.dot(w,mu_)*a1

        x_mlp_gat, x_gat, x_gcn, class_pair_intra, class_pair_inter, intra_gammas_mlp_gat, inter_gammas_mlp_gat, intra_gammas_gat, inter_gammas_gat = all_gammas(Xw, adjlist, row1, row2, row3, row4, a1, a2, b)

        pred = np.zeros(n)
        pred[x_mlp_gat > 0] = 1
        tmp = 1 - np.sum(np.abs(pred - ground_truth))/n
        acc_mlp_gat_list.append(tmp)
        
        pred = np.zeros(n)
        pred[x_gat > 0] = 1
        tmp = 1 - np.sum(np.abs(pred - ground_truth))/n
        acc_gat_list.append(tmp)
        
        pred = np.zeros(n)
        pred[x_gcn > 0] = 1
        tmp = 1 - np.sum(np.abs(pred - ground_truth))/n
        acc_gcn_list.append(tmp)
        
        pred = np.zeros(n)
        pred[Xw > 0] = 1
        tmp = 1 - np.sum(np.abs(pred - ground_truth))/n
        acc_lin_list.append(tmp)
        
        sum_intra_avg_mlp_gat += np.asarray(intra_gammas_mlp_gat).mean()
        sum_intra_std_mlp_gat += np.asarray(intra_gammas_mlp_gat).std()
        sum_inter_avg_mlp_gat += np.asarray(inter_gammas_mlp_gat).mean()
        sum_inter_std_mlp_gat += np.asarray(inter_gammas_mlp_gat).std()
        
        for head in range(heads):
            sum_intra_avg_gat[head][0] += np.asarray(intra_gammas_gat[head][0]).mean()
            sum_intra_std_gat[head][0] += np.asarray(intra_gammas_gat[head][0]).std()
            sum_inter_avg_gat[head][0] += np.asarray(inter_gammas_gat[head][0]).mean()
            sum_inter_std_gat[head][0] += np.asarray(inter_gammas_gat[head][0]).std()
            
            sum_intra_avg_gat[head][1] += np.asarray(intra_gammas_gat[head][1]).mean()
            sum_intra_std_gat[head][1] += np.asarray(intra_gammas_gat[head][1]).std()
            sum_inter_avg_gat[head][1] += np.asarray(inter_gammas_gat[head][1]).mean()
            sum_inter_std_gat[head][1] += np.asarray(inter_gammas_gat[head][1]).std()
        
        pairs_intra_list.append(class_pair_intra)
        pairs_inter_list.append(class_pair_inter)

    intra_avg_mlp_gat.append(sum_intra_avg_mlp_gat/trials)
    intra_std_mlp_gat.append(sum_intra_std_mlp_gat/trials)
    inter_avg_mlp_gat.append(sum_inter_avg_mlp_gat/trials)
    inter_std_mlp_gat.append(sum_inter_std_mlp_gat/trials)
    
    for head in range(heads):
        intra_avg_gat[head][0].append(sum_intra_avg_gat[head][0]/trials)
        intra_std_gat[head][0].append(sum_intra_std_gat[head][0]/trials)
        inter_avg_gat[head][0].append(sum_inter_avg_gat[head][0]/trials)
        inter_std_gat[head][0].append(sum_inter_std_gat[head][0]/trials)
        
        intra_avg_gat[head][1].append(sum_intra_avg_gat[head][1]/trials)
        intra_std_gat[head][1].append(sum_intra_std_gat[head][1]/trials)
        inter_avg_gat[head][1].append(sum_inter_avg_gat[head][1]/trials)
        inter_std_gat[head][1].append(sum_inter_std_gat[head][1]/trials)
    
    acc_mlp_gat.append(np.asarray(acc_mlp_gat_list).mean())
    acc_gat.append(np.asarray(acc_gat_list).mean())
    acc_gcn.append(np.asarray(acc_gcn_list).mean())
    acc_lin.append(np.asarray(acc_lin_list).mean())
    
    acc_pairs_intra.append(np.asarray(pairs_intra_list).mean())
    acc_pairs_inter.append(np.asarray(pairs_inter_list).mean())
    
    acc_mlp_gat_std.append(np.asarray(acc_mlp_gat_list).std())
    acc_gat_std.append(np.asarray(acc_gat_list).std())
    acc_gcn_std.append(np.asarray(acc_gcn_list).std())
    acc_lin_std.append(np.asarray(acc_lin_list).std())
    
    acc_pairs_intra_std.append(np.asarray(pairs_intra_list).std())
    acc_pairs_inter_std.append(np.asarray(pairs_inter_list).std())
        

fig = plt.figure(figsize=(11, 7), dpi=80)

plt.plot(mus*(2*np.sqrt(d)),intra_avg_mlp_gat, linewidth=2, linestyle='-', marker='*', markersize=9,  label='Average $\gamma$, intra edges, MLP-GAT')
plt.fill_between(mus*(2*np.sqrt(d)),np.asarray(intra_avg_mlp_gat)-np.asarray(intra_std_mlp_gat),np.asarray(intra_avg_mlp_gat)+np.asarray(intra_std_mlp_gat),alpha=0.4)

plt.plot(mus*(2*np.sqrt(d)),inter_avg_mlp_gat, linewidth=2, linestyle='-', marker='X', markersize=9, label='Average $\gamma$, inter edges, MLP-GAT')
plt.fill_between(mus*(2*np.sqrt(d)),np.asarray(inter_avg_mlp_gat)-np.asarray(inter_std_mlp_gat),np.asarray(inter_avg_mlp_gat)+np.asarray(inter_std_mlp_gat),alpha=0.4)

gammas_gcn = (1/(1+0.5*n*(p+q)))*np.ones(len(mus))
    
plt.plot(mus*(2*np.sqrt(d)),gammas_gcn, linewidth=2, linestyle='-', marker='o', markersize=9, color = 'black', label='Average $1/|N_i|$')

# idx  = [i for i,x in enumerate(acc_lin) if x==1][0]
idx2 = find_nearest(mus*(2*np.sqrt(d)),std_)
# plt.axvline(x=mus[idx]*(2*np.sqrt(d)), color='green', linewidth=4, linestyle='--', label='100% accuracy')
plt.axvline(x=mus[idx2]*(2*np.sqrt(d)), color='purple', linewidth=4, linestyle='-.', label='$\sigma$')

plt.grid(linestyle='dashed')
plt.legend(fontsize=20)
plt.xscale('log')
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.xlabel('Distance between means', fontsize=20)
plt.ylabel('$\gamma$ value', fontsize=20)
plt.show()
# fig.savefig("figures/gammas_MLPGAT_varying_distance_q_01.pdf", dpi=400, bbox_inches='tight')        