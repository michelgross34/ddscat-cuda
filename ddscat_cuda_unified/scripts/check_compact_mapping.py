# Pure layout test for the CUDA compact gather/scatter indexing.
import numpy as np
nat=10
occupied=np.array([0,2,5,9],dtype=np.int64)
nat0=len(occupied)
full=(np.arange(3*nat,dtype=np.float32)+1j*np.arange(3*nat,dtype=np.float32)/10).astype(np.complex64)
compact=np.empty(3*nat0,dtype=np.complex64)
for q in range(3*nat0):
    c=q//nat0; k=q-c*nat0; j=int(occupied[k])
    compact[q]=full[c*nat+j]
back=np.zeros_like(full)
for q in range(3*nat0):
    c=q//nat0; k=q-c*nat0; j=int(occupied[k])
    back[c*nat+j]=compact[q]
expected=np.zeros_like(full)
for c in range(3): expected[c*nat+occupied]=full[c*nat+occupied]
assert np.array_equal(back,expected)
assert compact.size==3*nat0
print('OK   compact component-major gather/scatter round trip')
print(f'OK   synthetic occupancy factor={nat0/nat:.6f}')
