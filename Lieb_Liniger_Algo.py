from numpy import arctan, zeros, pi, real as re, imag as im, linspace,eye, prod, newaxis
from numpy import array as arr, exp, log, arange, diag, kron, savetxt, cumsum, argmax
from numpy.linalg import det, norm, solve
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from copy import deepcopy
from time import clock
from numpy.random import uniform, shuffle
from scipy import integrate
import matplotlib.patches as pat
import pylab as pl

   
def ncr(n,r):
    if r < 0:
        return 0
    p,q = 1,1
    for j in range(r):
        p *= n-j
        q *= j+1
    return p//q

def ind_draw(o,m,n): #generator giving all ways to draw n elements from range(o,m) without replacement
    if n>=0:
        l = m-o
        if n == 0:
            yield []
        elif n == l:
            yield list(range(o,m))
        else:
            for k in range(o,m-n+1):
                for wha in ind_draw(k+1,m,n-1):
                    yield [k]+wha

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
    return w


def boxicles(n,deps): #puts n particles in boxes with maximal capacity deps. oo=None was last input
    M = len(deps)
    if n == 0:
        yield [0 for _ in range(M)]
    else:
        for preput in boxicles(n-1,deps):
            for k in range(M):
                #postput = [a for a in preput]
                #postput[k] += 1
                #yield postput
                if preput[k]<deps[k]:
                    yield [preput[a]+int(a==k) for a in range(M)]
                if preput[k]:
                    break

def TBA(T,c,chempot,givedens = True):
    interpts = 101 #odd is best
    bdys = 20
    la = linspace(-bdys,bdys,interpts) #lambda
    dla = la[1]-la[0]
    ootp = 1/(2*pi)
    lala = (la*la-chempot)/T
    convo = dla*c/(pi*(c*c+(la[:,newaxis]-la[newaxis,:])**2))
    tba = lambda eps: lala- eps +(convo*log(1+exp(-eps))[newaxis,:]).sum(axis=1)
    exep = exp(fsolve(tba,zeros(interpts)))
    ooexep = 1/(1+exep)
    #plt.plot(la,ooexep)
    tba2 = lambda rhop: rhop/(ootp+(convo*rhop[newaxis,:]).sum(axis=1))-ooexep
    rhopsol =  fsolve(tba2,0.15*ooexep)
    #plt.plot(la,ooexep/rhopsol)
    rhopsol -= min(rhopsol) #ensure non-negativity, despite numerical error
    D = sum(rhopsol)*dla
    if givedens:
        return D
    else:
        rhot = ootp+(convo*rhopsol[newaxis,:]).sum(axis=1)
        xi = [0]
        for rj in rhot:
            xi.append(xi[-1]+dla*rj)
        xi = (arr(xi[1:])+arr(xi[:-1]))/2
        xi -= xi[interpts//2]
        return rhopsol/rhot, xi
        
def LL_gibbs(N,L,T,c,ss): #more closely recreates a gibbs ensemble from a finite set of states.
    qngen = LL_thermal_disc(N,L,T,c,200*ss)
    ensemble = []
    pref = 2*pi/L
    for qn in qngen:
        aqn = arr(qn)
        lam,_ = newtrap(aqn,L,c,aqn*pref)
        ensemble.append([sum(lam*lam),qn])
    ensemble.sort()
    h=1
    while h<len(ensemble):
        if ensemble[h-1][0]==ensemble[h][0]:
            if ensemble[h-1][1]==ensemble[h][1]:
                ensemble.pop(h)
        h += 1
    energies = arr([e[0] for e in ensemble])
    prolly = cumsum(exp(-energies/T))
    prolly /= prolly[-1]
    
    #plt.hist(energies,bins=linspace(0,150,100))
    #plt.plot(prolly)
    for _ in range(ss):
        yield ensemble[argmax(prolly>uniform())][1]


def LL_thermal_disc(N,L,T,c,samplesize):
    if N==0:
        for _ in range(samplesize):
            yield []
    else:
        dens = N/L
        chempot = fsolve(lambda chemp: TBA(T,c,chemp)-dens,1)
        #plt.plot([TBA(T,c,ch) for ch in linspace(-10,10,100)])
        rhox,xi = TBA(T,c,chempot,False)
        pref = 1/L
        #dom = max(1000,T/L)
        #print(xi[0],xi[-1])
        nf = lambda k : smirt(xi,rhox,k)
        #KX = linspace(-10,10,1000)
        #plt.plot(KX,[nf(kx) for kx in KX])
        #boundbox = int(fsolve(lambda bd: integrate.quad(nf,-bd*pref,bd*pref)[0]-0.99*dens,L/2)[0]+2) #find the Qn inside which 99.5% of the particles should be
        boundbox = int(xi[-1]*L)
        for _ in range(samplesize):
            if N%2:
                I = [0]
                index = 1
            else:
                I = []
                index = 0.5
            sign  = 1
            newreject = []
            while len(I) < N and index<boundbox:
                ki = index*pref
                if uniform()<nf(ki):
                    I.append(sign*index)
                else:
                    newreject.append(sign*index)
                if sign == 1:
                    sign = -1
                else:
                    sign = 1
                    index += 1
            while len(I) < N:
                shuffle(newreject)
                reject = newreject
                shuffle(reject)
                rejlen,rejind = len(reject),0
                newreject = []
                while len(I) < N and rejind<rejlen:
                    if uniform()<nf(pref*reject[rejind]):
                        I.append(reject[rejind])
                    else:
                        newreject.append(reject[rejind])
                    rejind +=1
            if uniform()<0.5:
                I = [-ii for ii in I]
            yield sorted(I)
            
def smirt(x,y,a): # y(x) irregularly spaced, x in increasing tho. a is desired x coordinate: interpolate
    n = len(y)-1
    h = 0
    if a<x[0] or a>x[-1]:
        return 0
    while x[h+1]<a and h<n:
        h += 1
    return y[h]+(y[h+1]-y[h])*(a-x[h])/(x[h+1]-x[h])


def fd_better_disc(N,L,T,samplesize):
    pref = 2*pi/L
    if L==0:
        for _ in range(samplesize):
            yield []
    else:
        beta = 0.5/T
        #dom = max(1000,T/L)
        dom = 100
        dens = 2*pi*N/L
        mu = fsolve(lambda moo: integrate.quad(lambda k: 1/(1+exp((k*k-moo)*beta)),-dom,dom)[0]-dens,0)[0]
        nf = lambda k : 1/(1+exp((k*k-mu)*beta))
        boundbox = int(fsolve(lambda bd: integrate.quad(nf,-bd*pref,bd*pref)[0]-0.99*dens,L/2)[0]+2) #find the Qn inside which 99.5% of the particles should be
        for _ in range(samplesize):
            if N%2:
                I = [0]
                index = 1
            else:
                I = []
                index = 0.5
            sign  = 1
            newreject = []
            while len(I) < N and index<boundbox:
                ki = index*pref
                if uniform()<nf(ki):
                    I.append(sign*index)
                else:
                    newreject.append(sign*index)
                if sign == 1:
                    sign = -1
                else:
                    sign = 1
                    index += 1
            while len(I) < N:
                shuffle(newreject)
                reject = newreject
                shuffle(reject)
                rejlen,rejind = len(reject),0
                newreject = []
                while len(I) < N and rejind<rejlen:
                    if uniform()<nf(pref*reject[rejind]):
                        I.append(reject[rejind])
                    else:
                        newreject.append(reject[rejind])
                    rejind +=1
            yield sorted(I)


def gaudmat2(lam,leng,c): #gaudin matrix, needed to solve for rapidities
    Ljk = lam[:,newaxis]-lam
    K = 2*c/(Ljk*Ljk+c*c)
    if len(lam)==0:
        return 1
    return eye(len(lam))*leng+diag(sum(K))-K

def bethLL2(lam,II,leng,ooc): #Lieb liniger repulsive, ooc = 1/c
    return 2*(arctan((lam[:,newaxis]-lam)*ooc).sum(axis=1))-2*pi*II+lam*leng

def newtrap(II,L,c,lamguess=False): #execution time scales linearly with number of iterations. Worth the reduction.
    if len(II)==0:
        return arr([]),1,0
    ooc = 1/c
    if type(lamguess) != bool:
        lam = lamguess + 0
    else:
        lam = arr([ai for ai in II],dtype=float)
    tol = 10**-13*len(II)**0.5
    res = bethLL2(lam,II,L,ooc)
    iters = 0
    gm = gaudmat2(lam,L,c)
    while norm(res)>tol and iters<100:
        lam += solve(gm,-res)
        res = bethLL2(lam,II,L,ooc)
        gm = gaudmat2(lam,L,c)
        iters += 1
    return lam,det(gm)#,iters

def getbackflows(lam,L,c):
    N = len(lam)
    return fsolve(dliv,eye(N).reshape(N*N),args=(lam,L,c)).reshape((N,N))
    
def dliv(G,lam,L,c):
    N = len(lam)
    gmat = G.reshape((N,N))
    al = lam-lam[:,newaxis]
    chi = 2*c/(c*c+al*al)
    return (gmat*(L+chi.sum(axis=1)[:,newaxis])-chi.dot(gmat)-2*pi*eye(N)).reshape(N*N)

def sashaLLdata(opi):
    samplesize = 50
    maxcomplex = 400000
    dopsi = False
    if dopsi:
        NLNL = [16,24]
        target = 0.98
    else:
        NLNL = [15]
        target = 12
    for T in [6]: #[1,3, 
        for NL in NLNL:#[13,17,21]:
            for c in [4]:
                qngen = LL_thermal_disc(NL,NL,T,c,10*samplesize)
                opi.append([T,NL,c,[]])
                h = 0
                rej = 0
                while h < samplesize:
                    qnn = next(qngen)
                    aaa = LLstate(NL,T,4,c,NL,dopsi=dopsi,qn=qnn)
                    if aaa.complexity < maxcomplex and aaa.complexity>0: #could have overflow
                        kloek = clock()
                        h += 1
                        print(h,end=': ')
                        aaa.prep()
                        aaa.hilbert_search(target)
                        print('seconds elapsed',int(clock()-kloek))
                        opi[-1][-1].append([qnn,aaa.states_done,deepcopy(aaa.operator)])
                        if not dopsi:
                            print('erows',len(aaa.operator[0]),'pcols',max([len(aaa.operator[0][j]) for j in range(len(aaa.operator[0]))]))
                    else:
                        rej += 1
                if not dopsi:
                    opi[-1].append((aaa.dP,aaa.dE,aaa.gsenergy))
                print('Done with',[T,NL,c],'rejected:',rej)
    if dopsi:
        avg = [arr([opi[k][3][j][2] for j in range(samplesize)]).sum(axis=0)/samplesize for k in range(len(opi))]
        return avg #run stringsdelete(opi,avg):
    #else:
    #    return opi

def getaxes(dPdEgsE,dfactor,ecut,DSF2,gslam):
    bump = 0#exp(-20)
    for j in range(1):
        workdsf = DSF2[j][:ecut[j],:]
        orishape = workdsf.shape
        boundsE = arr([-dPdEgsE[2],-dPdEgsE[2]+orishape[0]*dPdEgsE[1]])/(gslam[-1]**2)
        boundsP =( orishape[1]/2*dPdEgsE[0])/gslam[-1]
        rebinned = log(rebin(workdsf,tuple([orishape[k]//dfactor[j][k] for k in [0,1]]))+bump)
        print(max([max(plu) for plu in rebinned]))
        print(rebinned.shape)
        EE = linspace(boundsE[0],boundsE[1],rebinned.shape[0])
        PP = linspace(-boundsP,boundsP,rebinned.shape[1])
        boxsize = (PP[1]-PP[0])*(EE[1]-EE[0])
        rebinned -= log(boxsize)
        plt.figure()
        plt.imshow(rebinned)
        plt.colorbar()
        # np.savetxt('./hydrostat.csv', np.column_stack(( np.reshape(X, (lx*ly,1)), np.reshape(Y, (lx*ly,1)), np.reshape(T1, (lx*ly,1))  )), header='x, y, T', comments='# ',delimiter=',', newline='\n' )
        #with open('DSF_T'+['1','3','6'][j]+'.dat', 'wb') as your_dat_file:  
        #    your_dat_file.write(struct.pack('i'*len(rebinned), *rebinned))
        #csvsurf('DSF_Tnew'+['1c','3c','6c'][j+2],rebinned,EE,PP)
        return PP,EE
    

def combinedsf(dsflist):
    maxE = 0
    maxP = 0
    for operator in dsflist:
        for half in operator:
            maxE = max(maxE,len(half))
            for erow in half:
                maxP = max(maxP,len(erow))
    total = zeros((maxE,2*maxP))
    for operator in dsflist:
        for pj,half in enumerate(operator):
            for ej,erow in enumerate(half):
                for pjj,pcol in enumerate(erow):
                    pk = (1-pj)*(maxP-1-pjj)+pj*(maxP+pjj)
                    total[ej,pk] += pcol
    return total/len(dsflist)
                    
def rebin(oldarr, new_shape):
    shop = (new_shape[0], oldarr.shape[0] // new_shape[0],new_shape[1], oldarr.shape[1] // new_shape[1])
    return oldarr.reshape(shop).sum(-1).sum(1)

def averageopi(taui):
    avtau = []
    xpts = 201
    place = linspace(0,10,xpts) #change back to symmetric -10,10?
    rpts = 5
    ray = linspace(0,0.4,rpts)
    raytime = kron(place,ray).reshape((xpts,rpts))
    for tau in taui:
        avtau.append(tau[:3])
        op = 0
        N = len(tau[3][0][0])
        Place = 1j*place[:,newaxis]
        Time = -0.5j*raytime 
        for state in tau[3]:
            lam,_ = newtrap(arr(state[0]),N,4,arr(state[0])*2*pi/N)
            energy = sum(lam*lam)
            momi = sum(lam)
            op += state[1]*exp(-momi*Place-energy*Time)
        avtau[-1].append(op/len(tau[3]))
    return avtau

def plotavopi(avopi):
    ru=0
    xx = linspace(0,10,201)
    cols = ['firebrick','darkorange','forestgreen','dodgerblue','purple']
    bgs = ['lightcoral','bisque','lightgreen','lightskyblue','plum']
    j=0
    for avo in avopi[ru::]:
        error = 1-abs(avo[3][0][0])
        plt.fill_between(xx, abs(avo[3][:,0])-error, abs(avo[3][:,0])+error,facecolor=bgs[j],edgecolor=cols[j],alpha=0.2)
        j+=1
    j=0
    for avo in avopi[ru::]:
        plt.plot(xx,abs(avo[3][:,0]),label='[T, N]='+str(avo[:2]),c=cols[j])
        j+=1
    plt.legend()
    plt.gcf().set_size_inches(5,4)
    plt.title('Thermal Lieb-Liniger '+r'$|\langle\psi(x)\psi(0)\rangle|; c=$'+str(avopi[0][2]))
    plt.xlabel('x')
    plt.ylabel(r'$|\langle\psi\psi\rangle|$')
    plt.savefig('PsipsiLLc'+str(avopi[0][2])+'.png',dpi=500)

def LLpsistats(L,T,ss,etarget,ptarget,res,pez,ez):
    erange = 1
    prange = 1
    c = 4
    B = 4
    wish = 0.9
    count = 0
    QNs = []
    qngen = fd_better_disc(L,L,T,1000000)
    while count < ss:
        qnow = next(qngen)
        if qnow not in QNs:
            QNs.append(qnow)
            aaa= LLstate(L,T,B,c,L,qn=qnow)
            mom = sum(aaa.lam)
            go = mom>ptarget-prange and mom<ptarget+prange and aaa.energy > etarget-erange and aaa.energy < etarget+erange
            if aaa.complexity<1000000 and go:
                count += 1
                print('count=',count)
                aaa.prep()
                aaa.hilbert_search(wish)
                res.append(deepcopy(aaa.operator))
                pez.append(deepcopy(aaa.qn))
                ez.append(sum(aaa.lam*aaa.lam))

def sashwrite(res,pez,ez,lab): #lab is str describing the experiment run
    savetxt("LLstash/"+'energies_'+lab+".CSV",arr(ez),delimiter =',')
    for h in range(len(ez)):
        savetxt("LLstash/"+'psi_'+lab+'_'+str(h)+".CSV",res[h],delimiter =',')
        savetxt("LLstash/"+'qns_'+lab+'_'+str(h)+".CSV",arr(pez[h]),delimiter =',')


class LLstate:
    #to do: vary the value of B, eps, to get to sumrule most quickly
    def __init__(self,N,T,B,c,L,dopsi = True, qn=False, eps=0.3, ceps=0.2, ftff = False): #take B=4 or 5?
        #eps and ceps are cutoffs, determine where to separate microshufflings and ph-profiles into tiers
        self.states_done = 0
        self.clo = clock()
        self.N = N
        self.L = L
        self.B = B
        if B >=N:
            print('box too large')
        self.T = T
        self.c = c
        if type(qn)==bool:
            self.qngen = LL_thermal_disc(N,L,T,c,100) #fd_better_disc(N,L,T,100) 
            self.qn = arr(next(self.qngen))
        else:
            self.qn = arr(qn)
        self.lam,self.detgl = newtrap(self.qn,L,c,self.qn*2*pi/L)
        self.energy = sum(self.lam*self.lam)
        self.momi = sum(self.lam)
        self.dens = N/L
        self.eps = eps #separate tiers by a factor of eps
        self.carteps = ceps #discourage exploration by a factor carteps
        self.peps = 2
        #self.stateslist = []
        self.dopsi = dopsi
        if (N%2) and dopsi:
            print('dont work for odd N')
        self.ftff = ftff
        if not ftff:
            self.maxpad = (N//B)*10 #how many boxes shall we look into Hilbert space?
            self.maxorder = N-dopsi
        else:
            self.maxpad = 1# was 2
            self.maxorder = 1
        self.sum_rule = 0
        #self.smell = 0
        self.get_box_filling()
        self.brackup = None
        self.saving = False
        self.prebrackup = []

    def prep(self,cutoff = 1000000,pri=True):
        if pri:
            print('boxfil:',self.boxfil,'shuf:',self.complexity)
        if self.dopsi:
            #we are doing the psi operator
            self.source,_ = newtrap(self.closeqn,self.L,self.c,self.closeqn*2*pi/self.L)
            
            self.np = int(500/log(self.c*self.N)) #auxilliary variables for handling multiplication
            self.ntp = ((self.N-1)*(self.N-2))//(2*self.np)+1
            

            xpts = 201
            place = linspace(0,10,xpts) #change back to symmetric -10,10?
            rpts = 5
            ray = linspace(0,1,rpts)
            raytime = kron(place,ray).reshape((xpts,rpts))
            self.Place = 1j*place[:,newaxis]
            self.Time = -1j*raytime 
            self.operator = zeros((xpts,rpts),dtype=complex)
            smallgirl = self.lam[:-1,newaxis]- self.lam[newaxis,:-1]
            #construct constants needed for calculating the operator later on
            self.preK = (2*self.c/(smallgirl*smallgirl+self.c*self.c) - (2*self.c/((self.lam[-1]-self.lam[:-1])**2+self.c*self.c))[newaxis,:])
            self.preK /= prod((self.lam[:,newaxis]-self.lam[newaxis,:]+eye(self.N)),axis=0)[:-1,newaxis]
            #prew = 1/detgl
            LJK = []#[0]
            for j in range(self.N):
                for k in range(j):
                    #ljk = self.lam[j]-self.lam[k]
                    LJK.append(self.lam[j]-self.lam[k])
                    #prew *= (ljk*ljk+c*c)*ljk*ljk
            al = arr(LJK)
            multthis = (al*al+self.c*self.c)*al*al
            self.prelog = sum(log(multthis)) - log(self.detgl)
            #have to do this shit with logarithms... fudge.
            #multthis[0] = 1/detgl
            #lelijk = frexp(multthis)
            #premantpre = prod(lelijk[0])
            #self.exppre = sum(lelijk[1])
            #self.mantpre,lel = frexp(premantpre)
            #self.exppre += lel
            #print(prod(lelijk[0])*2.**(sum(lelijk[1])),prew)
            #self.mantpre,self.exppre = frexp(prew)
            #self.prew=prew
            self.Vjbot = 1/prod(self.lam[:,newaxis]- self.lam[newaxis,:-1]+self.c*1j,axis=0)
        else:
            #now calculates DSF, or rho rho operator
            self.source = self.lam
            if not self.ftff:
                ppts = 20
                epts = 2
                self.operator = [[],[]]
                gsqn = arange((1-self.N)/2,self.N/2)
                gslam,_ = newtrap(gsqn,self.L,self.c,gsqn*2*pi/self.L)
                self.gsenergy =sum(gslam*gslam)
                self.dE = 1.5*self.gsenergy/epts
                self.dP = self.N/ppts
            broadlam = self.lam[:,newaxis]- self.lam[newaxis,:]+0j
            prodbroad = (broadlam+ 1j*self.c).prod(axis=0)
            self.VJbot = 1/prodbroad
            #self.Vbotprod = prod(prodbroad)
            cocobot = (broadlam+eye(self.N)).prod(axis=0)
            self.c0bot = re(cocobot)
            self.Kjk = 2*self.c/(broadlam*broadlam+self.c*self.c)-(2*self.c/(self.lam*self.lam+self.c*self.c))[newaxis,:]
            self.VPbot = 1/prod(self.lam+1j*self.c)
            #self.branorm = abs(prod(self.c0bot)/(self.detgl*self.Vbotprod)) # end was c0bot)*(1j**-(L%4)))
            self.loglamnorm = 0.5*(sum(log(abs(prodbroad)))+sum(log(abs(cocobot)))-log(abs(self.detgl))) #not just norm #((-1j)**self.N)*
            self.powahs = arr([1]*(self.N+2)+[0.5]*self.N+[-1]*(self.N+1)+[-0.5]*(self.N+1))
            self.sss = ((self.N*(self.N-1))//2)%2
        if len(self.source)==0:
            self.GBF = arr([[]])
        else:
            self.GBF = getbackflows(self.source,self.L,self.c)
        
        if self.complexity < cutoff and self.complexity>0: #second is against overflow
            self.ground_shuffle()
            self.brackup = None
            self.saving = False
            self.build_block(0,0)             

    def update_operator(self,ket,flow):
        #check components one by one, using for loops.
        if self.brackup:
            mus,detgm = next(self.brackup)
        else:
            mus,detgm = newtrap(arr(sorted(ket)),self.L,self.c,flow)
            if self.saving:
                self.prebrackup.append([mus,detgm])
        e = sum(mus*mus) -self.energy
        p = sum(mus) -self.momi
        if self.dopsi:
            mulan = mus[:,newaxis]-self.lam[newaxis,:]
            pmu= prod(mulan)
            imVJ = 2*im(prod(mus[:,newaxis]-self.lam[newaxis,:-1]+1j*self.c,axis=0)*self.Vjbot)
            detU = det(diag(imVJ)+ prod(mulan[:,:-1],axis=0)[:,newaxis]*self.preK)
            mjk = []
            for j in range(self.N-1):
                for k in range(j):
                    mjk.append(mus[j]-mus[k])
            mjk = arr(mjk)
            mjk2 = mjk*mjk
            mjk2c = mjk2+self.c*self.c
            logm = sum(log(arr([prod([mjk2[j*self.np:(j+1)*self.np]]) for j in range(self.ntp)])))
            logmc = sum(log(arr([prod([mjk2c[j*self.np:(j+1)*self.np]]) for j in range(self.ntp)])))
            lofi = log(arr([abs(detgm),abs(pmu),abs(detU)])) #,dtype=complex: need to reinstate phase
            w = exp(0.5*(self.prelog+2*lofi[2]+logm-logmc-lofi[0]-2*lofi[1]))
            w2 = w*w
            self.operator += w2*exp(p*self.Place+e*self.Time)
        else:
            if abs(p)<3e-14:
                w=0
            else:
                mulanvec= (mus-self.lam)
                broadmus = mus[newaxis,:]- mus[:,newaxis] #transposing this results in -1 on N=19
                #mubot = (broadmus+eye(self.N)).prod(axis=0)
                #ketnorm = prod(mubot)/(detgm*prod(prodbroad))#*(1j**-((self.L-self.dopsi)%4)) #don't need phase.
                
                mulan = mus[:,newaxis]- self.lam[newaxis,:]
                imVJ = 2*im(prod((mulan+1j*self.c),axis=0)*self.VJbot) #correct
                Vp = prod(mus+1j*self.c)*self.VPbot #correct
                
                utop = (mulan-diag(mulanvec)+eye(self.N)).prod(axis=0)
                U = eye(self.N)+(utop*mulanvec/(imVJ*self.c0bot))[:,newaxis]*self.Kjk
                #ff = det(U)*sum(mulanvec)*prod(imVJ)*self.Vbotprod/(prod(mulan)*2*im(Vp)) #was 1j*imVJ and #*2j*im(Vp)
                #w = ff*(self.branorm*ketnorm)**0.5 #they all have phase -pi/4 or 3pi/4.. why?
                #print(w,ff*ketnorm**0.5*exp(0.5*self.prelog))
                
                hoglog = [det(U),p]
                hoglog.extend(imVJ)
                hoglog.extend((broadmus+eye(self.N)).prod(axis=0))
                hoglog.append(2*im(Vp))
                hoglog.extend(prod(mulan,axis=0))
                hoglog.append(detgm)
                hoglog.extend((broadmus.T+ 1j*self.c).prod(axis=0))
                
                #pluslog = [det(U)+0j,sum(mulanvec)]
                #pluslog.extend(imVJ)
                #print('pluslog',len(pluslog))
                
                #halfpluslog=(broadmus+eye(self.N)).prod(axis=0)+0j #norm mus
                #print('halfpluslog',len(halfpluslog))
                
                #minlog = [2*im(Vp)+0j] #norm mus
                #minlog.extend(prod(mulan,axis=0))
                #print('minlog',len(minlog))
                
                #halfminlog = [detgm]
                #halfminlog.extend((broadmus+ 1j*self.c).prod(axis=0)) #norm mus
                #print('halfminlog',len(halfminlog))
    

                sisi = [1,-1][((abs(2*sum(sum((mulan<0)))-self.N*self.N)-self.N)//2)%2]
                
                #w = exp(sum(log(pluslog))-sum(log(minlog))+0.5*(sum(log(halfpluslog))-sum(log(halfminlog)))+self.loglamnorm)
                #(2*(det(broadmus)>0)-1)*(2*(det(broadmus)>0)-1)*(2*(prod(mulanvec)<0)-1)
                w = sisi*exp(sum(self.powahs*log(abs(arr(hoglog))))+self.loglamnorm)
                #w = (1j)**(abs(2*sum(sum((mulan<0)))-self.N*self.N)-self.N)*exp(sum(self.powahs*log(abs(arr(hoglog))))+self.loglamnorm)
                #if w==0:
                #    print('p',abs(p))
            w2 = w*w
            #w2 = abs(ff*self.branorm*ff*ketnorm) #ff.conjugate()
            if not self.ftff:
                self.dsf(w2,p,e)
        #ML = '' #show ordering of lambdas and mus
        #for mulu in sorted([[moo,'M'] for moo in mus]+[[loo,'L'] for loo in self.lam]):
        #    ML += mulu[1]
        #print(ML,w)
        self.sum_rule += w2
        self.states_done +=1
        return w,w2 #wont work when bra=ket: divergencies.

    def qn2box(self,q):
        return 2*abs(q//self.B)-int(q<0)
    
    def box2qn(self,boxj): #gives the first qn of the box of index boxj
        return (1-2*(boxj%2))*((boxj+1)//2)*self.B

    def dsf(self,w,p,e):
        sp = int(p>0)
        indp = int(abs(p)/self.dP)
        inde = int((e+self.gsenergy)/self.dE)#increased energy by one gs energy just to ensure its positive.
        le = len(self.operator[sp])-1
        if inde > le:
            for _ in range(inde-le):
                self.operator[sp].append([])
        lp = len(self.operator[sp][inde])-1
        if indp > lp:
            for _ in range(indp-lp):
                self.operator[sp][inde].append(0.)
        self.operator[sp][inde][indp] += w
    
    def plotdsf(self,Vmin=-37):
        fig, ax = pl.subplots(figsize=(10, 10))
        ML1 = []
        for opera in self.operator:
            ML1.append(max([len(ener) for ener in opera]))
        ML0 = max(len(self.operator[0]),len(self.operator[1]))
        for hp in [0,1]:
            self.operator[hp].extend([[] for _ in range(ML0-len(self.operator[hp]))])
        img = []
        for hm0 in range(ML0-1,-1,-1):
            img.append([0 for _ in range(ML1[0]-len(self.operator[0][hm0]))])
            img[-1].extend(reversed(self.operator[0][hm0]))
            img[-1].extend(self.operator[1][hm0])
            img[-1].extend([0 for _ in range(ML1[1]-len(self.operator[1][hm0]))])
        self.img = arr(img)
        exten = [-self.dP*ML1[0],self.dP*ML1[1],0,ML0*self.dE]
        ax.imshow(log(self.img),aspect='auto',cmap=plt.cm.get_cmap('YlGnBu'),extent=exten,vmin=Vmin,vmax=0)
        ax.set_ylabel('E')
        ax.set_xlabel('P')
        ax.set_title('DSF, L='+str(self.L)+', N='+str(self.N)+', c='+str(self.c)+', T='+str(self.T))
        
    def sector_one(self,pri = True):
        maxp = len(self.p_h_profiles[1][-1][0][0])
        posis = [self.box2qn(j)//self.B+maxp//2 for j in range(maxp)]
        hosis = [self.innerboxes//2-1-self.box2qn(j)//self.B for j in range(self.innerboxes)]
        self.img = zeros((self.innerboxes,maxp))
        for pad in self.p_h_profiles[1]:
            for PES in pad:
                gp,gh = PES[0].index(1),PES[0].index(-1)
                rp = self.B
                if gp < self.innerboxes:
                    rp -= self.boxfil[gp]
                self.img[hosis[gh],posis[gp]] = PES[2]/(self.boxfil[gh]*rp) #should account for dI/d\lam in LL
        if pri:
            ax = plt.gca()
            ax.imshow(log(abs(self.img)),aspect='auto',extent=[-maxp//2,maxp//2,-self.innerboxes//2,self.innerboxes//2]) #was log...
            for hi,I in enumerate(range(-maxp//2,maxp//2)):
                for hj, J in enumerate(range(-self.innerboxes//2,self.innerboxes//2)):
                    if self.img[hj,hi]<0:
                        ax.add_patch(pat.Rectangle((I,-J-0.99),1,0.99,hatch='//',fill=False,snap=False))
            #plt.imshow(1/(self.img>0), cmap='binary', alpha=0.5,aspect='auto',extent=[-maxp//2,maxp//2,-self.innerboxes//2,self.innerboxes//2])
            plt.title('Log of FT operator (shaded is negative)')
            plt.xlabel('Box Particle')
            plt.ylabel('Box Hole')


    def get_box_filling(self):
        self.innerboxes = 2*max([abs(hu//self.B)+int(hu>=0) for hu in [int(self.qn[0]),int(self.qn[-1])]]+[(self.N+self.B-1)//(2*self.B)+1])
        self.boxfil = [0 for _ in range(self.innerboxes)]
        if self.dopsi: #to fill with one particle fewer.
            qpop = self.qn[min(list(range(self.N)),key= lambda fh: abs(self.qn[fh]))]
            #print(self.qn)
            #print(qpop)
            closeqn = []
            for q in self.qn:
                if q != qpop:
                    self.boxfil[self.qn2box(int(q))] += 1
                    closeqn.append(int(q))
            self.closeqn = arr(closeqn)
        else:
            for q in self.qn:
                self.boxfil[self.qn2box(q)] += 1
            self.closeqn = self.qn
        self.complexity = prod([ncr(self.B,bb) for bb in self.boxfil])
        self.boxemp = [self.B-bb for bb in self.boxfil]
        s = 0
        self.U = [[]]
        if not self.dopsi:
            self.diagind  = []
        for j in range(self.innerboxes):
            locshuf = []
            for nowind,nowbox in enumerate(ind_draw(s,s+self.B,self.boxfil[j])):
                locshuf.append(nowbox)
                if (not self.dopsi) and len(self.diagind)==j:
                    isitit = True
                    for q in nowbox:
                        if q not in self.qn:
                            isitit = False
                            break
                    if isitit:
                        self.diagind.append(nowind)
            self.U[0].append(locshuf)
            if s<0:
                s = -s #the start of the next box
            else:
                s = -s-self.B
            
    def ground_shuffle(self):
        #all inbox shuffling, determines the tiers of the unexcited boxes
        flatpf = [0 for _ in range(self.innerboxes)]
        #elements are [[profile],[extremal indices]]
        weightrix = zeros(shape=tuple([ncr(self.B,nnf) for nnf in self.boxfil]))
        walking = 0
        for ket,flow,multind,dyes in self.in_box_shuffle(self.boxfil,self.leftof(flatpf)): #final index is internal.
            if dyes: #this is recursively traced to only be True when ket and bra are the same state
                #print('bop')# check that the diagonal state is only once.
                p,e = sum(self.lam),sum(self.lam*self.lam)
                w = self.N/self.L #when bra==ket, FF diverges, choose operator artificially.
                w2 = w*w
                if not self.ftff:
                    self.dsf(w,p,e)
                self.sum_rule += w2
                self.states_done +=1
            else:
                w,w2 = self.update_operator(ket,flow)
            weightrix[tuple(multind)] = w2
            walking += w
        odw = disect(weightrix)
        self.BT = [[None for boj in range(self.innerboxes)]]
        self.p_h_profiles = [[[[flatpf,[],walking]]]] #indices [level (excitation)][pad][nr extremal indices] and
        self.sortout(odw,0,0)
        self.blocks = [[[]]]
        self.cart = []

    
    def explore(self,level,pad): #don't call with pad =0 and level = 0
        if level:
            b0 = self.innerboxes+2*pad
            flat = [0 for _ in range(b0)]
            self.spind = [None,None]#special index for filling in weight matrices
            self.spox = [None,None]
            if len(self.blocks[level-1]) <= pad:
                self.explore(level-1,pad)
            if pad:
                if level <= self.B:
                    partconf = ncr(self.B,level)
                    weightrix = [[0 for _ in range(partconf)] for _ in [0,0]]
                    for boxj in range(b0-2,b0):
                        s = self.box2qn(boxj)
                        self.U[level].append(list(ind_draw(s,s+self.B,level)))
                        self.BT[level].append([self.parasite(level,boxj)])
                #then create the ph profiles
                #self.p_h_profiles[level].append([])
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
                                for ket,flow in self.shuffle_profile_tier(profile,flat,self.leftof(profile)):
                                    w,w2 = self.update_operator(ket,flow)
                                    running += w2
                                    walking += w
                                    count += 1
                                    if self.spox[0] != None: #theres an extremal particle
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
                            nou.append(list(ind_draw(s,s+self.B,self.boxfil[boxj]+nl)))
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
                        for ket,flow in self.shuffle_profile_tier(profile,flat,self.leftof(profile)):
                            w,w2 = self.update_operator(ket,flow)
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
                    #carteps will discourage going to the edge: choosing the outermost pads, 
                    #because exploring has high computational cost
                    break
        else:
            self.BT[0].extend([[[[]]],[[[]]]])
            self.blocks[0].append([[]])
            self.boxfil.extend([0,0])

    def build_block(self,l,p):
        if len(self.blocks[l]) == p+1 and p<self.maxpad:
            self.explore(l,p+1)
            print(l,p+1,end='... ')
        if len(self.blocks) == l+1 and l<self.maxorder and p >= self.peps*l: #last demand is to suppress exploration of high levels.. now jogged demand.
            for p0 in range(self.maxpad+1):
                print(l+1,p0,end='... ')
                self.explore(l+1,p0)
                if self.cart and self.cart[-1][1]==[l+1,p0,0]:
                    return
            print('not enough tiers')

    
    def stack_block(self,l,p,t):
        if len(self.blocks[l][p]) == t+1:
            #print('higher:',l,p,t)
            self.blocks[l][p].append(self.deepen_gen(l,p,t+1))
            cm = None
            for rm,cm in self.blocks[l][p][t+1]:
                if cm:
                    self.cart.append([rm/cm,[l,p,t+1]])
                    break
            

    def harvest(self): #not good, goes to higher l too quickly... lot of time wasted in l=2 for high pads...
        l,p,t = self.cart[0][1]
        if self.ftff: #Field Theory Form Factors: do all ph-profiles so we get fair representation.
            for _,_ in self.blocks[l][p][t]:
                pass
        else: #only do the most weight-bearing ph-profiles.
            if len(self.cart)>1:
                thresh = self.cart[1][0]
            else:
                thresh=0
            for rm,cm in self.blocks[l][p][t]:
                if rm < thresh*cm:
                    self.cart[0][0] = rm/cm
                    return
        self.cart.pop(0)
    
    def hilbert_search(self,wish,maxtime = 0): #wish as a proportion of the sumrule 1
        if not maxtime:
            self.clo = self.N*10 + clock()
        else:            
            self.clo = maxtime + clock()
        while self.sum_rule < wish*self.dens and clock()<self.clo:
            if self.cart:
                self.cart.sort(reverse=True)
                l,p,t = self.cart[0][1]
                #print(l,p,t,end='...')
                lc = len(self.cart)
                self.build_block(l,p)
                self.stack_block(l,p,t)
                if len(self.cart)> lc:
                    continue
                self.harvest()
            else:
                print('this should never happen')
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
                    print('!!')
                    return
        print('!')
                
    def leftof(self,profile):
        s = len(profile)
        LIBOS = [None for p in profile]
        l = 0
        g = s-1
        while g != s:
            LIBOS[g] = l
            l += self.boxfil[g]+profile[g]
            if g == 1:
                g = 0
            elif (g%2):
                g -= 2
            else:
                g += 2
        return LIBOS
            
    def shuffle_profile_tier(self,phpf,tierpf,lebolist): #list of where the particles and holes are, where the tier excitations are
        #accomodates backflows.
        #lebolist is the number of particles to the left of the current box, needed to calculate backflows.
        l = len(phpf)
        if l==0:
            yield [],self.source
        else:
            for tail,preflow in self.shuffle_profile_tier(phpf[:-1],tierpf[:-1],lebolist[:-1]):
                ribo = lebolist[-1] + phpf[-1] + self.boxfil[l-1]
                shift = self.closeqn[lebolist[-1]:ribo]
                derivs = self.GBF[lebolist[-1]:ribo]
                for nowbox in self.BT[phpf[-1]][l-1][tierpf[-1]]:
                    #print(lebolist[-1],nowbox,shift)
                    #print(preflow+(nowbox-shift).dot(derivs))
                    #calculate postflow with global positions of qns...
                    yield tail + nowbox, preflow+(nowbox-shift).dot(derivs)

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
        for PES in self.p_h_profiles[level][pad]:
            running = 0
            walking = 0
            count = 0
            for tierpf in self.tier_profile(PES[0],PES[1],tier):
                for ket,flow in self.shuffle_profile_tier(PES[0],tierpf,self.leftof(PES[0])):
                    count += 1
                    w,w2 = self.update_operator(ket,flow)
                    running += w2
                    walking += w
            PES[2] += walking
            yield running,count
        
    def sortout(self,odw,nl,pad):
        if pad:
            bn = self.innerboxes+2*(pad-1) #boxnow
        else:
            bn = 0
        #tjek = []
        for w in odw:
            argord = (-w).argsort()
            noweps = 1
            bt = []
            for j in argord:
                if not (w[j] > noweps): #if had overflow earlier in calculating weight, then w is nan and this doesn't satisfy
                    noweps *= self.eps
                    bt.append([]) #so then this never gets appended and bt is an empty list
                bt[-1].append(self.U[nl][bn][j]) #and this fails.
            self.BT[nl][bn] = bt
            bn += 1
            #tjek.append(len(bt)-1)
        #if not sum(tjek):
        #    print('no higher tiers',self.boxfil)
    
        
    def in_box_shuffle(self,Bfil,lebolist):# returns qn,backflow,indices,and boolean whether we are building the diagonal state
        l = len(Bfil)
        if l == 0:
            yield [],self.source,[],not self.dopsi
        else:
            for tail,preflow,indis,dyes in self.in_box_shuffle(Bfil[:-1],lebolist[:-1]):
                j=0
                ribo = lebolist[-1]+ self.boxfil[l-1]
                shift = self.closeqn[lebolist[-1]:ribo]
                derivs = self.GBF[lebolist[-1]:ribo]
                for nowbox in self.U[0][l-1]:# use self.U
                    yield tail + nowbox, preflow+(nowbox-shift).dot(derivs),indis+[j],dyes and (j==self.diagind[l-1]) #must enter the indis as a tuple
                    j += 1

