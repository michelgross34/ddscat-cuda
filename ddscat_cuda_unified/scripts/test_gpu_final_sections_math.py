#!/usr/bin/env python3
import numpy as np
rng=np.random.default_rng(12345)
N=257
r=rng.normal(size=(N,3))
p=(rng.normal(size=(N,3))+1j*rng.normal(size=(N,3))).astype(np.complex128)
k=rng.normal(size=3); k*=2.3/np.linalg.norm(k); n=k/np.linalg.norm(k)
phase=np.exp(-1j*(r@k))
# Original SCAT form: project every dipole, then sum.
orig=np.sum((p-n[None,:]*np.sum(p*n[None,:],axis=1)[:,None])*phase[:,None],axis=0)
# GPU form: sum raw P first, then one transverse projection.
s=np.sum(p*phase[:,None],axis=0)
gpu=s-n*np.dot(n,s)
assert np.allclose(orig,gpu,rtol=2e-13,atol=2e-13),np.max(np.abs(orig-gpu))
# Selected direction: sum (P dot em)*phase equals em dot sum(P*phase).
em=rng.normal(size=3); em-=n*np.dot(n,em); em/=np.linalg.norm(em)
orig_f=np.sum((p@em)*phase)
gpu_f=np.dot(em,s)
assert np.allclose(orig_f,gpu_f,rtol=2e-13,atol=2e-13)
# EVALQ algebra for a symmetric 3x3 inverse polarizability block.
ad=rng.normal(size=(N,3))+1j*rng.normal(size=(N,3))
ao=rng.normal(size=(N,3))+1j*rng.normal(size=(N,3)) # [a23,a31,a12]
k3=1.7
q=np.empty_like(p)
q[:,0]=(ad[:,0]+1j*k3/1.5)*p[:,0]+ao[:,1]*p[:,2]+ao[:,2]*p[:,1]
q[:,1]=(ad[:,1]+1j*k3/1.5)*p[:,1]+ao[:,2]*p[:,0]+ao[:,0]*p[:,2]
q[:,2]=(ad[:,2]+1j*k3/1.5)*p[:,2]+ao[:,0]*p[:,1]+ao[:,1]*p[:,0]
z=np.vdot(p.reshape(-1,order='F'),q.reshape(-1,order='F'))
z2=sum(np.vdot(p[j],q[j]) for j in range(N))
assert np.allclose(z,z2,rtol=2e-13,atol=2e-13)
print('PASS: SCAT transverse-projection equivalence')
print('PASS: selected-direction amplitude equivalence')
print('PASS: EVALQ symmetric-block algebra / complex dot equivalence')
