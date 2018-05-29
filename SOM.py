import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SOM:
    def __init__(self, x=25, y=25):
        self.shape = (x, y) 
        self.neurons = np.random.random_sample((x, y, 3))
        self.input = np.random.random_sample((x + y, 3))

    def train(self):
        self.ims = []
        self.iteration = 0
        for neuron in self.input:
            self.iteration += 1

            sigma0 = max(self.shape[0], self.shape[1]) / 2
            alpha = len(self.input)/np.log(sigma0) 
            sigma = sigma0*np.exp(-self.iteration/alpha)

            distances = np.full(self.neurons.shape, neuron) 
            dist = np.vectorize(np.linalg.norm) 
            distances = dist(self.neurons - distances)
            center = np.unravel_index(distances.argmin(), distances.shape)
            center = np.array([center[0], center[1]])
            
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    dist = np.linalg.norm(center - np.array([i, j])) 
                    if (dist < sigma):
                        L0 = 0.8
                        L = L0 * np.exp(-self.iteration/alpha)
                        Theta = np.exp(-(dist**2/sigma**2))
                        self.neurons[i, j] = self.neurons[i, j] + Theta * L * (neuron - self.neurons[i, j])
                        im = plt.imshow(self.neurons, cmap='hot', animated=True)
                        self.ims.append([im])


if __name__ == '__main__':
    som = SOM()
    som.train()

    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, som.ims, interval=1, blit=True, repeat=False)
    plt.show()