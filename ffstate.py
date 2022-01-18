# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:38:48 2019

@author: Daniel

"""
from numpy import zeros, pi, array as arr, sin, exp, prod, linspace, newaxis, kron
from numpy.linalg import det
import matplotlib.pyplot as plt
from time import clock

def ind_draw(o,m,n): 
    if n>=0:
        if n == 0:
            yield []
        else:
            for k in range(o,m-n+1):
                for wha in ind_draw(k+1,m,n-1):
                    yield [k]+wha

def boxicles(n,depths): #puts n particles in boxes with maximal capacity deps. oo=None was last input
    M = len(depths)
    if n == 0:
        yield [0 for _ in range(M)]
    else:
        for preput in boxicles(n-1,depths):
            for k in range(M):
                #postput = [a for a in preput]
                #postput[k] += 1
                #yield postput
                if preput[k]<depths[k]:
                    yield [preput[a]+int(a==k) for a in range(M)]
                if preput[k]:
                    break

def ncr(n,r):
    if r < 0:
        return 0
    p,q = 1,1
    for j in range(r):
        p *= n-j
        q *= j+1
    return p//q

def disect(We): #We is l-dimensional tensor, returns the sum along all but each axes.
    w = []
    As = We.shape
    l = len(As)
    for j in range(l):
        if As[j] >1:
            w.append(We.sum(tuple(range(1,l-j))))
        else:
            w.append(arr([1]))
        We = We.sum(0)
    for j in range(l):
        if As[j] >1: #don't calculate the full sum rule over and over again on trivial boxes. (empty or full)
            w[j] /= We
    return w#,We#[sorted(list(range(len(w[j]))),key=(-w[j]).__getitem__) for j in range(l)]



class FFstate:
    def __init__(self,qn,phi,B,L,eps=0.3,ceps=0.2,maxcomplex = 200000):
        self.states_done = 0
        self.clo = clock()
        self.N = len(qn)
        self.L = L
        self.B = B
        if B >=self.N:
            print('box too large')
        self.phi = phi
        self.maxorder = self.L
        self.qn = qn
        self.abra = arr(self.qn)[:,newaxis]
        shiftqn = arr(self.qn)-phi
        self.pbra,self.ebra = -sum(shiftqn),-sum(shiftqn*shiftqn)
        self.presin = (sin(pi*self.phi)/pi)**self.N
        self.eps = eps
        self.carteps = ceps
        self.peps = 1
        sca = 2*pi/self.L
        xpts = 201
        place = linspace(0,10,xpts)
        rpts = 5
        ray = linspace(0,1,rpts)
        raytime = kron(place,ray).reshape((xpts,rpts))
        self.sum_rule = 0
        self.Place = sca*1j*place[:,newaxis]
        self.Time = -sca*sca*0.5j*raytime #
        self.tau = zeros((xpts,rpts),dtype=complex)
        self.get_box_filling()
        print('box fill:',self.boxfil,'shuffle states:',self.complexity,end=', ')
        if self.complexity < maxcomplex and self.complexity>0:
            self.ground_shuffle()
            self.build_block(0,0)
        else:
            print('This would take too long.')
        
    def qn2box(self,q):
        return 2*abs(q//self.B)-int(q<0)
    
    def box2qn(self,boxj):
        return (1-2*(boxj%2))*((boxj+1)//2)*self.B

    def get_local_shuffle(self,fillin,s):
        locshuf = []
        for bf in ind_draw(s,s+self.B,fillin):
            p,e = 0,0
            for n in bf:
                p += n
                e += n*n
            locshuf.append([bf,p,e])
        return locshuf

    def get_box_filling(self):
        self.innerboxes = 2*max([abs(hu//self.B)+int(hu>=0) for hu in 
            [self.qn[0],self.qn[-1]]]+[(self.N+self.B-1)//(2*self.B)+1])
        self.boxfil = [0 for _ in range(self.innerboxes)]
        self.maxpad = (self.N//self.B)*5
        for q in self.qn:
            self.boxfil[self.qn2box(q)] += 1
        self.complexity = prod([ncr(self.B,bb) for bb in self.boxfil])
        self.boxemp = [self.B-bb for bb in self.boxfil]
        s = 0
        self.U = [[]]
        for j in range(self.innerboxes):
            self.U[0].append(self.get_local_shuffle(self.boxfil[j],s))
            if s<0:
                s = -s
            else:
                s = -s-self.B

    def ground_shuffle(self):
        weightrix = zeros(shape=tuple([ncr(self.B,nnf) for nnf in self.boxfil]))
        walking = 0
        for ket,p,e,multind in self.in_box_shuffle(self.boxfil):
            w,w2 = self.update_tau(ket,p,e)
            weightrix[tuple(multind)] = w2
            walking += w
        odw = disect(weightrix)
        self.BT = [[None for boj in range(self.innerboxes)]]
        self.sortout(odw,0,0)
        self.p_h_profiles = [[[[[0 for _ in range(self.innerboxes)],[],walking]]]] 
        self.blocks = [[[]]]
        self.cart = []

    def stack_block(self,l,p,t):
        if len(self.blocks[l][p]) == t+1:
            self.blocks[l][p].append(self.deepen_gen(l,p,t+1))
            cm = None
            for rm,cm in self.blocks[l][p][t+1]:
                if cm:
                    self.cart.append([rm/cm,[l,p,t+1]])
                    break

    def harvest(self): 
        l,p,t = self.cart[0][1]
        if len(self.cart)>1:
            thresh = self.cart[1][0]
        else:
            thresh=0
        for rm,cm in self.blocks[l][p][t]:
            if rm < thresh*cm:
                self.cart[0][0] = rm/cm
                return
        self.cart.pop(0)
    
    def build_block(self,l,p):
        if len(self.blocks[l]) == p+1 and p<self.maxpad:
            self.explore(l,p+1)
            print(l,p+1,end=' ... ')
        if len(self.blocks) == l+1 and l<self.maxorder and p >= self.peps*l:
            for p0 in range(self.maxpad+1):
                print(l+1,p0,end=' ... ')
                self.explore(l+1,p0)
                if self.cart and self.cart[-1][1]==[l+1,p0,0]:
                    return
            print('not enough tiers')
    
    def hilbert_search(self,wish,maxtime = 0):
        if not maxtime:
            self.clo = self.N*3 + clock()
        else:            
            self.clo = maxtime + clock()
        while self.sum_rule < wish and clock()<self.clo:
            if self.cart:
                self.cart.sort(reverse=True)
                l,p,t = self.cart[0][1]
                lc = len(self.cart)
                self.build_block(l,p)
                self.stack_block(l,p,t)
                if len(self.cart)> lc:
                    continue
                self.harvest()
            else:
                l1 = len(self.blocks)
                for l in range(l1):
                    pma = len(self.blocks[l])
                    if pma <= self.maxpad:
                        self.explore(l,pma)
                        l = -1
                        break
                if l1 <= self.maxorder:
                    self.explore(l1,0)
                elif l>0:
                    return
        print('!')


    def explore(self,level,pad): 
        if level:
            b0 = self.innerboxes+2*pad
            flat = [0 for _ in range(b0)]
            self.spind = [None,None]
            self.spox = [None,None]
            if len(self.blocks[level-1]) <= pad:
                self.explore(level-1,pad)
            if pad:
                if level <= self.B:
                    partconf = ncr(self.B,level)
                    weightrix = [[0 for _ in range(partconf)] for _ in [0,0]]
                    for boxj in range(b0-2,b0):
                        self.U[level].append(self.get_local_shuffle(level,self.box2qn(boxj)))
                        self.BT[level].append([self.parasite(level,boxj)])
                profnweight = []
                for oparts in range(1,min(level,2*self.B)+1):
                    for opl in range(max(0,oparts-self.B),min(oparts,self.B)+1):
                        opr = oparts-opl
                        for partplace in boxicles(level-oparts,self.boxemp+[self.B for _ in range(2*(pad-1))]):
                            holesdeps = []
                            for j in range(self.innerboxes):
                                if partplace[j]:
                                    holesdeps.append(0)
                                else:
                                    holesdeps.append(self.boxfil[j])
                            for holeplace in boxicles(level,holesdeps):
                                profile = partplace+[opl,opr]
                                running = 0
                                walking = 0
                                count = 0
                                for jj in range(self.innerboxes):
                                    profile[jj] -= holeplace[jj]
                                for ket,p,e in self.shuffle_profile_tier(profile,flat):
                                    w,w2 = self.update_tau(ket,p,e)
                                    running += w2
                                    walking += w
                                    count += 1
                                    if self.spox[0] != None:
                                        weightrix[self.spox[0]%2][self.spind[0]] += w2
                                if self.spox[0] == None:
                                    extholes = []
                                else:
                                    extholes = [self.spox[0]]
                                    self.spox[0]=None
                                profnweight.append([running/max(1,count),[profile,extholes,walking]])
                self.p_h_profiles[level].append([xp for _,xp in sorted(profnweight,reverse=True)])
                if level <= self.B:
                    odw = []
                    for jw in [0,1]:
                        sw = sum(weightrix[jw])
                        odw.append(arr(weightrix[jw])/sw)
                    self.sortout(odw,level,pad)
                self.blocks[level].append([self.deepen_gen(level,pad,0)])
            else: #not pad: original boxes
                #self.p_h_profiles.append([])
                profnweight = []
                #first stock self.U
                if level <= self.B:
                    weightrix = []
                    for nl in [-level,level]:
                        weightrix.append([[0 for _ in range(ncr(self.B,bb-nl))] for bb in self.boxfil])
                        nou = []
                        self.BT.insert(level,[[self.parasite(nl,boj)] for boj in range(self.innerboxes)])
                        s=0
                        for boxj in range(self.innerboxes):
                            nou.append(self.get_local_shuffle(self.boxfil[boxj]+nl,s))
                            if s<0:
                                s = -s #the start of the next box
                            else:
                                s = -s-self.B
                        self.U.insert(level,nou)
                #then create ph profiles
                #self.p_h_profiles[level].append([]) #could also have extremal holes
                for partplace in boxicles(level,self.boxemp):
                    holesdeps = []
                    for j in range(self.innerboxes):
                        if partplace[j]:
                            holesdeps.append(0)
                        else:
                            holesdeps.append(self.boxfil[j])
                    for holeplace in boxicles(level,holesdeps):
                        profile = [partplace[jj]-holeplace[jj] for jj in range(self.innerboxes)]
                        running = 0
                        walking = 0
                        count = 0
                        for ket,p,e in self.shuffle_profile_tier(profile,flat):
                            w,w2 = self.update_tau(ket,p,e)
                            running += w2
                            walking += w
                            count += 1
                            for hw in [0,1]:
                                if self.spox[hw] != None:
                                    weightrix[hw][self.spox[hw]][self.spind[hw]] += w2
                        extholes = []
                        for sw in [0,1]:
                            if self.spox[sw]!= None:
                                extholes.append(self.spox[sw])
                                self.spox[sw] = None
                        profnweight.append([running/max(1,count),[profile,extholes,walking]])
                self.p_h_profiles.append([[xp for _,xp in sorted(profnweight,reverse=True)]])
                if level <= self.B:
                    for jw in [0,1]:
                        odw = []
                        for jww in range(self.innerboxes):
                            sw = sum(weightrix[jw][jww])
                            odw.append(arr(weightrix[jw][jww])/sw)
                        self.sortout(odw,[level,-level][jw],0)
                self.blocks.append([[self.deepen_gen(level,0,0)]])
            for rm,cm in self.blocks[level][pad][0]:
                if cm:
                    self.cart.append([rm/cm*self.carteps,[level,pad,0]]) #first value: predicted score.
                    #eps will discourage going to the edge: choosing the outermost pads, 
                    #because exploring has high computational cost
                    break
        else:
            self.BT[0].extend([[[[[],0,0]]],[[[[],0,0]]]])
            self.blocks[0].append([[]])

    def update_tau(self,ket,p,e):
        detti = det(1/(self.abra-sorted(ket)-self.phi))
        w = detti*self.presin 
        w2 = w*w
        self.tau += w2*exp(p*self.Place+e*self.Time)
        self.sum_rule += w2
        self.states_done +=1
        return w,w2


    
                
    def shuffle_profile_tier(self,phpf,tierpf): #list of where the particles and holes are, where the tier excitations are
        l = len(phpf)
        if l==0:
            yield [],self.pbra,self.ebra
        else:
            for tail,p,e in self.shuffle_profile_tier(phpf[:-1],tierpf[:-1]):
                for locshuf,p0,e0 in self.BT[phpf[-1]][l-1][tierpf[-1]]:
                    yield tail+locshuf,p+p0,e+e0


    def parasite(self,nl,boxj):
        ind = int(nl<0)
        self.spox[ind]=boxj
        for j,locfil in enumerate(self.U[nl][boxj]):
            self.spind[ind]=j
            yield locfil
        self.BT[nl][boxj][0] = self.parasite(nl,boxj) #lay its eggs again.
    
    def tier_profile(self,profile,EJ,tier): #treat it as tier+1, start at 0
        deps = [len(self.BT[profile[j]][j])-1 for j in range(len(profile))]
        if len(EJ)==1:
            deps[EJ[0]] = min(deps[EJ[0]],tier)
        if len(EJ)==2:
            a,b = deps[EJ[0]],deps[EJ[1]]
            deps[EJ[0]] = min(a+b,tier)
            deps[EJ[1]] = 0
            for tierpf in boxicles(tier+1,deps):
                if tierpf[EJ[0]] > a:
                    tierpf[EJ[1]] = tierpf[EJ[0]] - a
                    tierpf[EJ[0]] = a
                yield tierpf
                while tierpf[EJ[1]] < b and tierpf[EJ[0]] > 0:
                    tierpf[EJ[1]] += 1
                    tierpf[EJ[0]] -= 1
                    yield  tierpf
        else:
            for tierpf in boxicles(tier+1,deps):
                yield tierpf
                
    def deepen_gen(self,level,pad,tier): #generator for each profile at a given level,pad,tier.
        for PES in self.p_h_profiles[level][pad]: #PES = [profile, EJ, walking], EJ is extremal positions
            walking = 0
            running = 0
            count = 0
            for tierpf in self.tier_profile(PES[0],PES[1],tier):
                for ket,p,e in self.shuffle_profile_tier(PES[0],tierpf):
                    count += 1
                    w,w2 = self.update_tau(ket,p,e)
                    running += w2
                    walking += w
            PES[2] += walking
            yield running,count
        
    def sortout(self,odw,nl,pad):
        if pad:
            a = self.innerboxes+2*(pad-1) #boxnow
        else:
            a = 0
        #tjek = []
        for w in odw:
            argord = (-w).argsort()
            noweps = 1
            bt = []
            for j in argord:
                if not (w[j] > noweps):
                    noweps *= self.eps
                    bt.append([])
                bt[-1].append(self.U[nl][a][j])
            self.BT[nl][a] = bt
            a += 1
            #tjek.append(len(bt)-1)
        #if not sum(tjek):
        #    print('no higher tiers',self.boxfil)
        
        
    def in_box_shuffle(self,Bfil):# leftb is the leftmost possible box index, Bfil is the filling of the boxes from there
        l = len(Bfil)
        if l == 0:
            yield [],self.pbra,self.ebra,[] #4th variable is the index of this box, 5th is the beginning of the box
        else:
            for leftside,p0,e0,indis in self.in_box_shuffle(Bfil[:l-1]):
                j=0
                for nowbox,p,e in self.U[0][l-1]:# use self.U 
                    yield leftside + nowbox, p0+p,e0+e,indis+[j] #must enter the indis as a tuple
                    j += 1

            



















        