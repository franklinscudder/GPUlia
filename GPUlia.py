from numba import cuda
import numpy as np
from time import time

import pygame

class Viewer:
    def __init__(self, update_func, display_size):
        self.update_func = update_func
        pygame.init()
        self.display = pygame.display.set_mode(display_size)
    
    def set_title(self, title):
        pygame.display.set_caption(title)
    
    def start(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            Z = self.update_func()
            surf = pygame.surfarray.make_surface(Z)
            self.display.blit(surf, (0, 0))

            pygame.display.update()

        pygame.quit()

@cuda.jit
def julia_kernel(zxs, zys, res, maxiter, R, cx, cy):
    pos = cuda.grid(1)
    
    cx, cy = cx[0], cy[0]
    R = R[0]
    
    ix = pos // zxs.size
    iy = pos % zxs.size
    i = 0
    
    zx = zxs[ix]
    zy = zys[iy]
    
    cuda.syncthreads()
    
    if ix < zxs.size and iy < zys.size:
        while i < maxiter and (zx ** 2) + (zy ** 2) < (R ** 2):
            i += 1
            xtemp = (zx ** 2) - (zy ** 2)
            zy = 2 * zx * zy + cy
            zx = xtemp + cx
            
        if i == maxiter:
            res[pos] = 0.0
        else:
            res[pos] = i
            

def make_julia_cuda(cx, cy):
    r = 10.0
    
    while r**2 - r >= (cx**2 + cy**2)**0.5:
        r -= 0.001
        
        if r < 0:
            raise RuntimeError("Initial R value too small!")
    
    r += 0.002
    
    X, Y = 1000, 1000
    maxiter = 100
    
    R = cuda.to_device(np.array([r]))
    CX = cuda.to_device(np.array([cx]))
    CY = cuda.to_device(np.array([cy]))
    zxs = cuda.to_device(np.linspace(-r,r,X))
    zys = cuda.to_device(np.linspace(-r,r,Y))
    res = cuda.to_device(np.zeros((X * Y), "uint16"))
    
    threadsperblock = 1024
    blockspergrid = ((X*Y) + (threadsperblock - 1)) // threadsperblock
    
    julia_kernel[blockspergrid, threadsperblock](zxs, zys, res, maxiter, R, CX, CY)
    cuda.synchronize()
    res = res.copy_to_host()
    
    return res.reshape((X, Y)).T
 
 
def make_julia_cuda_next():
    t0 = time()
    global i
    
    c1 = 0.7*np.exp((i/212.2523252)*1j)
    c2 = 0.8*np.exp((i/130.23124)*1j)
    c3 = 0.2*np.exp((i/87.3453)*1j)

    c = c1 - c2 - c3
    
    i += 1
    cx = c.real
    cy = c.imag
    out = make_julia_cuda(cx, cy)
    print(int(1/(time() - t0)), c)
    return out
            
if __name__ == "__main__":
    i = 0
    v = Viewer(make_julia_cuda_next,(1000,1000))
    v.start()
