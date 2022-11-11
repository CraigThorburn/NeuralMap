# %% Import Statements
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation

# %% Define Variables
a = 0.04
num_cells = 500
L_start = 35
L_end = 1
L_decay = 200
num_trials = 1500

dim1_max = 2200
dim1_min = 200
dim2_max = 3300
dim2_min = 300

cat1_dim1_mean = 1000 #1000
cat1_dim1_var = 40 #40

cat1_dim2_mean = 1300 #1300
cat1_dim2_var = 60 #60

cat2_dim1_mean = 1000 #1000
cat2_dim1_var = 40 #40

cat2_dim2_mean = 2000 #2000
cat2_dim2_var = 60 #60

cat1_ratio = 0.5 #0.5
cat2_ratio = 0.5 #0.5

test_plts_x = np.array(
    [1200, 1375, 1550, 1725, 1900, 2075, 1200, 1375, 1550, 1725, 1900, 2075, 1200, 1375, 1550, 1725, 1900, 2075])
test_plts_y = np.array(
    [800, 800, 800, 800, 800, 800, 1000, 1000, 1000, 1000, 1000, 1000, 1200, 1200, 1200, 1200, 1200, 1200])


# %% Define Functions
def calculate_L(t):
    return L_end + L_start * math.exp(-t/400)

def formants2feats(x, dim_max, dim_min):
    return (x-dim_min)/(math.sqrt((x-dim_min)**2+(dim_max-x)**2)), (dim_max-x)/(math.sqrt((x-dim_min)**2+(dim_max-x)**2))

def feats2formants_plus(y, dim_max, dim_min):
    inv_y = 1-1/y**2
    return ((2*dim_max + 2*dim_min*inv_y) + math.sqrt((-2*dim_max -2*dim_min*inv_y)**2 -4*(2-1/y**2)*(dim_max**2 + dim_min**2*inv_y)))/(2*(2-1/y**2))

def feats2formants_minus(y, m, n):
    return (-math.sqrt((-m**2 * y**4) + (m**2 * y**2) + (2 * m * n * y**4) - (2 * m * n * y**2) - (n**2 * y**4) + (n**2 * y**2)) + (m * y**2) - m + n * y**2)/(2 * y**2 - 1)
    #(math.sqrt((-(m ** 2) * (y ** 4)) + ((m ** 2) * (y ** 2)) + (2 * m * n * y ** 4) - (2 * m * n * y ** 2) - (n ** 2 * y ** 4 + n ** 2 * y ** 2)) + m * y ** 2 - m + n * y ** 2) / ((2 * y ** 2) - 1)


def calculate_warping():

    test_warped_x = np.zeros(len(test_plts_x))
    test_warped_y = np.zeros(len(test_plts_x))
    for i in range(len(test_plts_x)):
        x1 = test_plts_y[i]
        x2 = test_plts_x[i]
        x1_plus = (x1 - dim1_min) / (math.sqrt((x1 - dim1_min) ** 2 + (dim1_max - x1) ** 2))
        x1_minus = (dim1_max - x1) / (math.sqrt((x1 - dim1_min) ** 2 + (dim1_max - x1) ** 2))
        x2_plus = (x2 - dim2_min) / (math.sqrt((x2 - dim2_min) ** 2 + (dim2_max - x2) ** 2))
        x2_minus = (dim2_max - x2) / (math.sqrt((x2 - dim2_min) ** 2 + (dim2_max - x2) ** 2))
        x = np.array([x1_minus, x1_plus, x2_minus, x2_plus])
        # print(x)
        activations = np.sum(x * zs, 1) / 2
        #print(activations)

        threshold_test = activations[np.argsort(activations)[-35]]
        m_test = np.array([ex if ex >= threshold_test else 0 for ex in activations])
        maxes = np.argsort(activations)[-35:]
        dim1_best = ((zs[:, 0] * dim1_min )+( zs[:, 1] * dim1_max)) / (zs[:, 0] + zs[:, 1])
        dim2_best = ((zs[:, 2] * dim2_min )+( zs[:, 3] * dim2_max)) / (zs[:, 2] + zs[:, 3])
        dim1_pop = np.mean(dim1_best[maxes])# sum(dim1_best * m_test) / sum(m_test)
        dim2_pop = np.mean(dim2_best[maxes])#sum(dim2_best * m_test) / sum(m_test)
        dim1_pop = sum(dim1_best * m_test) / sum(m_test)
        dim2_pop = sum(dim2_best * m_test) / sum(m_test)

        test_warped_x[i] = dim2_pop
        test_warped_y[i] = dim1_pop

    return test_warped_x, test_warped_y


    #plt.close()
    #fig = plt.figure()
    #plt.scatter(test_plts_x, test_plts_y, color='blue')
    #plt.scatter(test_warped_x, test_warped_y, color='red')

    #plt.xlim(1000, 2200)
    #plt.ylim(600, 1400)
    #plt.show()

def animate(j):
    i=j*2
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.scatter(test_plts_x, test_plts_y, color='blue', alpha=0.1, marker='o', s=350)
    ax1.scatter(warping_outputs[0][i], warping_outputs[1][i], color = 'red', s=350)

    ax2.scatter(presented_xs[i], presented_ys[i], color='green')
    ax2.scatter(percieved_xs[i], percieved_ys[i], color='black')
    ax3.scatter(dim2_bests[i], dim1_bests[i], color='black')
    ax1.set_xlim(1000,2250)
    ax1.set_ylim(650, 1300)
    ax2.set_xlim(1000,2250)
    ax2.set_ylim(650, 1300)
    ax3.set_xlim(1000,2250)
    ax3.set_ylim(650, 1300)

    #draw lines
    for x in range(5):
        for y in range(3):
            ax1.plot([warping_outputs[0][i][y*6+x], warping_outputs[0][i][y*6+x+1]], [warping_outputs[1][i][y*6+x],
                        warping_outputs[1][i][y*6+x+1]],linestyle = '-', color='red')

    for y in range(2):
        for x in range(6):
            ax1.plot([warping_outputs[0][i][y*6+x], warping_outputs[0][i][(y+1)*6+x]], [warping_outputs[1][i][y*6+x],
                        warping_outputs[1][i][(y+1)*6+x]],linestyle = '-', color='red')


    for x in range(5):
        for y in range(3):
            ax1.plot([test_plts_x[y*6+x], test_plts_x[y*6+x+1]], [test_plts_y[y*6+x],
                        test_plts_y[y*6+x+1]],linestyle = '-', color='blue', alpha = 0.1)

    for y in range(2):
        for x in range(6):
            ax1.plot([test_plts_x[y*6+x], test_plts_x[(y+1)*6+x]], [test_plts_y[y*6+x],
                        test_plts_y[(y+1)*6+x]],linestyle = '-', color='blue', alpha = 0.1)


    ax1.set_title('Iterations: '+str(i)+', Perceptual Map')
    ax2.set_title('Presented Stimuli')
    ax3.set_title('All Cells')
    ax1.set_xlabel('F3 Value')
    ax1.set_ylabel('F2 Value')
    ax2.set_ylabel('F2 Value')
    ax3.set_xlabel('F3 Value')
    ax3.set_ylabel('F2 Value')

    #print(presented_xs[i], presented_xs[i])
    #print(i)

# %% Define xs
print('defining inputs')
xs_cat1_dim1 = np.random.normal(cat1_dim1_mean, cat1_dim1_var, int(num_trials))
xs_cat1_dim2 = np.random.normal(cat1_dim2_mean, cat1_dim2_var, int(num_trials))
xs_cat2_dim1 = np.random.normal(cat2_dim1_mean, cat2_dim1_var, int(num_trials))
xs_cat2_dim2 = np.random.normal(cat2_dim2_mean, cat2_dim2_var, int(num_trials))


cats = [1 for x in range(int(num_trials*cat1_ratio))] + [2 for y in range(int(num_trials*cat2_ratio))]
random.shuffle(cats)

cat1 = 0
cat2 = 0
xs = np.zeros((num_trials,2))
for i in range(len(cats)):
    if cats[i]==1:
        xs[i] = np.array([xs_cat1_dim1[cat1], xs_cat1_dim2[cat1]])
        cat1+=1
    if cats[i]==2:
        xs[i] = np.array([xs_cat2_dim1[cat2], xs_cat2_dim2[cat2]])
        cat2+=1


print('inputs defined')
# %% Train
print('intiializing zs')
warping_outputs = ([], [])

zs_dim1 = np.random.rand(1, num_cells)*(dim1_max-dim1_min)+dim1_min
zs_dim2 = np.random.rand(1, num_cells)*(dim2_max-dim2_min)+dim2_min

zs_dim1_minus= (dim1_max-zs_dim1)/(np.sqrt((zs_dim1-dim1_min)**2+(dim1_max-zs_dim1)**2))
zs_dim1_plus = (zs_dim1-dim1_min)/(np.sqrt((zs_dim1-dim1_min)**2+(dim1_max-zs_dim1)**2))

zs_dim2_minus = (dim2_max-zs_dim2)/(np.sqrt((zs_dim2-dim2_min)**2+(dim2_max-zs_dim2)**2))
zs_dim2_plus = (zs_dim2-dim2_min)/(np.sqrt((zs_dim2-dim2_min)**2+(dim2_max-zs_dim2)**2))
zs = np.transpose(np.concatenate([ zs_dim1_minus, zs_dim1_plus, zs_dim2_minus, zs_dim2_plus ],0))

#zs = np.random.random((num_cells,4))

presented_xs = []
presented_ys = []
percieved_xs = []
percieved_ys = []
dim1_bests = []
dim2_bests = []

print('training model')
for t in range(len(xs)) :

    L = round(calculate_L(t))
    x1 = xs[t, 0]
    x2 = xs[t, 1]

    x1_plus = (x1-dim1_min)/(math.sqrt((x1-dim1_min)**2+(dim1_max-x1)**2))
    x1_minus = (dim1_max-x1)/(math.sqrt((x1-dim1_min)**2+(dim1_max-x1)**2))
    x2_plus = (x2-dim2_min)/(math.sqrt((x2-dim2_min)**2+(dim2_max-x2)**2))
    x2_minus= (dim2_max-x2)/(math.sqrt((x2-dim2_min)**2+(dim2_max-x2)**2))
    x = np.array([ x1_minus, x1_plus, x2_minus, x2_plus ])
    #print(x)
    activations = np.sum(x*zs,1)/2
    threshold = activations[np.argsort(activations)[-L]]
    m = np.array([ex if ex>=threshold else 0 for ex in activations ])
    #print(m)
    difs = x-zs
    deltas = np.array([a * m[i] * difs[i] for i in range(len(difs))])
    zs = zs + deltas
    #print(x)

    threshold_test = activations[np.argsort(activations)[-35]]
    m_test = np.array([ex if ex >= threshold_test else 0 for ex in activations])
    dim1_best = ( zs[:,0] * dim1_min + zs[:,1] * dim1_max)/(zs[:,0] + zs[:,1])
    dim2_best = (zs[:, 2] * dim2_min + zs[:, 3] * dim2_max) / (zs[:, 2] + zs[:, 3])
    maxes = np.argsort(activations)[-35:]
    dim1_pop = np.mean(dim1_best[maxes]) #sum(dim1_best * m_test)/sum(m_test)
    dim2_pop = np.mean(dim2_best[maxes]) #sum(dim2_best * m_test)/sum(m_test)
    #threshold = activations[np.argsort(activations)[-35]]
    #m = np.array([ex if ex >= threshold else 0 for ex in activations])
    #print(m)
    #F = np.array([ zs[i] * m[i] for i in range(len(m))])/np.sum(m)
    #print(F)
    #print('input:', x1, '->', dim1_pop, '    ', x2, '->', dim2_pop)
    #z = input()
    warping_outs_x, warping_outs_y = calculate_warping()
    warping_outputs[0].append(warping_outs_x)
    warping_outputs[1].append(warping_outs_y)

    presented_xs.append(x2)
    presented_ys.append(x1)
    percieved_xs.append(dim2_pop)
    percieved_ys.append(dim1_pop)
    dim1_bests.append(dim1_best)
    dim2_bests.append(dim2_best)


#print(presented_xs)
#print(presented_ys)
#print(percieved_xs)
#print(percieved_xs)

print('training complete')
# %% Graph
print('plotting simulation')
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(6)
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=2, rowspan=3)
ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 2))
ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 2))
ax4 = plt.subplot2grid(shape=(3, 3), loc=(2, 2))

#fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ani = FuncAnimation(fig, animate, frames=750, interval=10, repeat=False)
ax4.hist(xs[:,1], bins=200)
ax4.set_title('Input Stimuli F3')
ax4.set_xlim(1000,2250)
ax4.set_xlabel('F3 Value')
ax4.set_ylabel('Count')
plt.tight_layout()
plt.show()


