# -*- coding: utf-8 -*-

import simplex
import numpy as np

def get_vector(s):
    l=s.split()
    v=[]
    for x in l:
        if '\\' in x:
            frac=x.split('\\')
            num=int(frac[0])
            den=int(frac[1])
            v.append(num/den)
        else:
            v.append(int(x))
    return v

cs=input('Digite o vetor de custo:\n')
ct=np.array([get_vector(cs)])

st=[]
print('Digite as restrições (pressione Ctrl+C após a última):')
while True:
    try:
        line = input()
    except KeyboardInterrupt:
        break
    st.append(line)
# restr=st.split('\n')
a=[]
b=[]
for x in st:
    vec=get_vector(x)
    a.append(vec[:-1])
    b.append([vec[-1]])
a=np.array(a)
b=np.array(b)

z,x,base=simplex.two_phase_simplex(a,b,ct)
if z is None:
    print('O problema é inviável.')
elif z==float('inf'):
    print('O problema é ilimitado.')
else:
    print('Valor ótimo da função objetivo: %f' % z)
    print('Solução ótima:')
    for i in range(np.shape(x)[0]):
        print('x%d = %f' % (i,x[i,0]))
    
    
