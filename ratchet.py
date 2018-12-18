from pylab import *
from scipy import integrate
from numpy import linalg as LA

# Discretization of theta
theta=arange(-pi,pi,0.001)

# Force, F(x) = -\grad sin(x+1/2*sin(x))
force=-(1.+cos(theta)/2.)*cos(theta + sin(theta)/2.)

# Equations of motion:
# \dot{x} = force + gamma + \sqrt{2 sigma} N(0,1)

# Size of the noise
sigma=0.3
# Size of the net drift
gamma=0.0

# Basis size 2M+1
M = 15
modes=arange(-M,M+1,1)

# Container for tilted operator
Ltilt=zeros([2*M+1,2*M+1])+1j*0

# Variable conjugate to the current
dlam1=0.05
lambdas=arange(-2,2.01,dlam1)
# Variable conjugate to the integrated force
dlam2=0.05
lambdas2=arange(-1.5,1.51,dlam2)

# Container for \psi(\lambda)
# Current large deviation function
psi=zeros([len(lambdas)])

# Container for \lambda2 dependent variance
kappa=[]
eta=[]

print "Lambda_1 Lambda_2 \psi(Lambda_1,Lambda_2)"

figure(figsize=(12,3))

output=open('psi.txt','w')

for jj in range(len(lambdas2)):
    lam2=lambdas2[jj]
    for ii in range(len(lambdas)):
        lam=lambdas[ii]
        for i in modes:
            for k in modes:
                # normalized Fourier basis Exp[inx]
                phii=(cos(i*theta)+1j*sin(i*theta))/sqrt(2*pi)
                phik=(cos(k*theta)-1j*sin(k*theta))/sqrt(2*pi)
                # L = (F(x)+gamma)(d/dx+\lambda_1)+\sigma (d/dx+\lambda_1)^2 - \lambda_2 F(x)
                drift=integrate.simps(phii*phik*((force+gamma)*(-1j*k - lam)+lam2*force),theta)
                diff=integrate.simps(sigma*(-k**2. + 2.*lam*1j*k + lam**2.)*phii*phik,theta)
                Ltilt[i,k]=drift+diff

        psis,v=LA.eig(Ltilt)
        psi[ii] = max([i for i in psis if abs(i.imag) < 1e-8]).real
        print lam, lam2, psi[ii]
        print >> output, lam, lam2, psi[ii]
    print >> output,'\n'

    subplot(131)
    plot(lambdas,psi-min(psi))
    meanj=(psi[1:]-psi[:-1])/dlam1
    subplot(132)
    plot(lambdas[1:]-dlam1/2.,meanj)
    varj=(meanj[1:]-meanj[:-1])/dlam1
    subplot(133)
    plot(lambdas[2:]-dlam1,varj)
    kappa=append(kappa,min(varj))
    
    thirdj=(varj[1:]-varj[:-1])/dlam1
    fourthj=(thirdj[1:]-thirdj[:-1])/dlam1
    eta=append(eta,min(fourthj))

subplot(131)
xlabel(r'$\lambda_1$',size=20)
ylabel(r'$\psi(\lambda_1|\lambda_2)$',size=18)
subplot(132)
xlabel(r'$\lambda_1$',size=20)
ylabel(r'$J_{\lambda_2}(\lambda_1)$',size=18)
subplot(133)
xlabel(r'$\lambda_1$',size=20)
ylabel(r'$\delta J^2_{\lambda_2}(\lambda_1)$',size=18)
tight_layout()
savefig('fig1.png')

zz,zzz=polyfit(lambdas2,kappa,2,cov=True)
yy,yyy=polyfit(lambdas2,eta,2,cov=True)

print
print "For \sigma= ", sigma
print "<J^2>: ", zz[2]
print "error: ", sqrt(zzz[2,2])
print "<J^2Q>: ",-zz[1]
print "error: ", sqrt(zzz[1,1])
print "<J^4>: ", yy[2]+3*zz[2]*zz[2]
print "error: ", sqrt(yyy[2,2])
print "<J^2Q^2>: ", zz[0]*2
print "error: ", sqrt(zzz[0,0])*2
print "First order response: ", zz[2]/2/sigma
print "Second order response: ", -zz[1]/4/sigma/sigma
print "Third order response: ", (zz[0]*2*3+(yy[2]+3*zz[2]*zz[2])-6*sigma*zz[2])/48/sigma/sigma/sigma
print

figure(2)
plot(lambdas2,kappa,'o')
plot(lambdas2,lambdas2*lambdas2*zz[0]+lambdas2*zz[1]+zz[2],'-r')
xlabel(r'$\lambda_2$',size=18)
ylabel(r'$\delta J^2_{\lambda_2}(0)$',size=18)
tight_layout()
savefig('fig2.png')

figure(3)
plot(lambdas2,eta,'o')
plot(lambdas2,lambdas2*lambdas2*yy[0]+lambdas2*yy[1]+yy[2],'-r')
xlabel(r'$\lambda_2$',size=18)
ylabel(r'$\delta J^4_{\lambda_2}(0)$',size=18)
tight_layout()
savefig('fig3.png')

