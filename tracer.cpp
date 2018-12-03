#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>	
#include <algorithm>	//std::copy
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <mpi.h>

using namespace std;

struct Walker_info{
	vector<double> x;
	vector<double> v;
	int parent;
	double Qavg;
};

double fRand(double fMin, double fMax){
	double f = (double)rand()/RAND_MAX;
	return fMin + f*(fMax-fMin);
}

double Ux(double x, double y, double U0, double L)
{
        double pi = 3.141592653589793238463;
        return U0*sin(2*pi*x/L)*cos(2*pi*y/L);
}

double Uy(double x, double y, double U0, double L)
{
        double pi = 3.141592653589793238463;
        return -U0*cos(2*pi*x/L)*sin(2*pi*y/L);
}

// this function takes the initial value x&v and modify it
// and returns the cumulative current Q within tint 
double propagation(double& x, double& y, double& vx, double& vy,double& q1, double tint, gsl_rng* rng)
{
        double h = 0.001;
        int Nstep = tint/h;
        int i,j;
	double vxnew,vynew;
	double fx,fy;
	double U0 = 1.0;
	double L = 1.0;
	double f = 0.0;
	double T = 0.0002;
	double gamma = 0.1;
	double a,b;
	double sum = 0.0;
	double xold,vxold,Uxold;

	a = exp(-gamma*h);
        b = sqrt(2/gamma/h*tanh(gamma*h/2));
        
        fx = gamma*Ux(x,y,U0,L) + f;
        fy = gamma*Uy(x,y,U0,L);
        
	xold = x;
	
        for(i=0;i<Nstep;i++){
		vxold = vx;
		Uxold = Ux(x,y,U0,L);

                vxnew = sqrt(a)*vx + sqrt((1-a)*T)*gsl_ran_gaussian(rng,1);
                vynew = sqrt(a)*vy + sqrt((1-a)*T)*gsl_ran_gaussian(rng,1);
                vxnew += b*h/2*fx; 
                vynew += b*h/2*fy;
                
                x += b*h*vxnew;
                y += b*h*vynew;
                
                fx = gamma*Ux(x,y,U0,L) + f;
                fy = gamma*Uy(x,y,U0,L);
                
                vxnew += b*h/2*fx;
                vynew += b*h/2*fy; 
                vx = sqrt(a)*vxnew + sqrt((1-a)*T)*gsl_ran_gaussian(rng,1);
                vy = sqrt(a)*vynew + sqrt((1-a)*T)*gsl_ran_gaussian(rng,1);

		sum += (vx-vxold) - gamma*Uxold*h;
        }

        q1 = x-xold;
        return sum;	
	}

int main()
{
        gsl_rng* rng;
        gsl_rng_env_setup();
        rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng,time(NULL));

	srand(time(0));
	
	//initializing the MPI environment
	MPI_Init(NULL,NULL);
	int nprocs,rank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Status status;	

	int i,j,k,s,index;
	
	//parameters for CANSS
	int Nw = 2400;
	int n = Nw/nprocs;
	double lambda1 = 0.0;
	double lambda2 = 0.0;
	double tint = 800.0, tobs = 800000.0;
	int Ntint = tobs/tint;
	double weightsum = 0.0;
	int walkersum = 0;
	vector<double> globalq(Nw),localq(n);
	vector<double> globalweight(Nw),localweight(n);
	double q1,q2,Q,Q2,localQ,localQ2,sigma;
	double phi = 0.0;

	vector<double> x(n);
	vector<double> y(n);
	vector<double> vx(n);
	vector<double> vy(n);
	vector<double> newx(n),newy(n),newvx(n),newvy(n);
	vector<double> weight(Nw);
	vector<int> number(Nw);
	vector<int> parent(Nw);
	vector<double> Qavg(Nw);
	vector<int> oldparent(Nw);
	vector<double> oldQavg(Nw);

	vector<double> mean,var,ldf;
	vector<int> multiplicity,m;
	vector<int> table;
	table.reserve(Nw);

	//Initialization of all the walkers
	for(i=0;i<n;i++){
		x[i] = 0.0;
		y[i] = 0.0;
		vx[i] = 0.0;
		vy[i] = 0.0;
	}

	for(i=0;i<Nw;i++){
		parent[i] = i;
		Qavg[i] = 0.0;
	}
	
	for(i=0;i<Ntint;i++){
		walkersum = 0;
		weightsum = 0.0;

		for(j=0;j<n;j++){
			q1 = 0.0;
			q2 = propagation(x[j],y[j],vx[j],vy[j],q1,tint,rng);
			localq[j] = q1;
			localweight[j] = lambda1*q1 + lambda2*q2;
		}	

		// Gather all the currents generated in tint to rank 0
		MPI_Gather(&localq[0],n,MPI_DOUBLE,&globalq[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gather(&localweight[0],n,MPI_DOUBLE,&globalweight[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// Reweight the walkers on rank 0 and write the lookup table 
		if(rank==0){
		for(j=0;j<Nw;j++){
			Qavg[j] += globalq[j];
			weight[j] = exp(-1*globalweight[j]);
			weightsum += weight[j];
		}
		for(j=0;j<Nw;j++){
			number[j] = floor(Nw*weight[j]/weightsum + fRand(0,1));
			walkersum += number[j];
		}
		if(walkersum < Nw){
			while(walkersum < Nw){
				number[fRand(0,Nw)] += 1;
				walkersum += 1;
			}	
		}
		if(walkersum > Nw){
			while(walkersum > Nw){
				s = floor(fRand(0,Nw));
				if(number[s]>0){
					number[s] -= 1;
					walkersum -= 1;
				}
			}
		}		 	
		for(j=0;j<Nw;j++){
			for(s=0;s<number[j];s++){
				table.push_back(j);
			}
		}
		}
					
		MPI_Bcast(&table[0],Nw,MPI_INT,0,MPI_COMM_WORLD);
		
		for(j=0;j<Nw;j++){
			if(table[j]!=j){
			//if the replaced walker is not on the same core
			if((table[j]/n)!=(j/n)){
			if(rank==table[j]/n){
				MPI_Send(&x[table[j]%n],1,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
				MPI_Send(&y[table[j]%n],1,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
				MPI_Send(&vx[table[j]%n],1,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
				MPI_Send(&vy[table[j]%n],1,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
			}
			if(rank==j/n){
				MPI_Recv(&newx[j%n],1,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);
				MPI_Recv(&newy[j%n],1,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);  
				MPI_Recv(&newvx[j%n],1,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);  
				MPI_Recv(&newvy[j%n],1,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);  
			}
			}
			else{
				if(rank==j/n){
					newx[j%n] = x[table[j]%n];
					newy[j%n] = y[table[j]%n];
					newvx[j%n] = vx[table[j]%n];
					newvy[j%n] = vy[table[j]%n];
				}
			}
			}
		}

		// Search in the lookup table for replacing walkers	
		for(j=0;j<n;j++){
			index = rank*n+j;
			if(table[index]!=index){
				x[index%n] = newx[index%n];
				y[index%n] = newy[index%n];
				vx[index%n] = newvx[index%n];
				vy[index%n] = newvy[index%n];
			}
		}

		// replace the Q's at rank 0 in the inverse order
		if(rank==0){
			Q = 0.0;
			Q2 = 0.0;
			for(j=0;j<Nw;j++){
				oldQavg[j] = Qavg[j];
				oldparent[j] = parent[j];
			}
			for(j=0;j<Nw;j++){
				if(table[j]!=j){
					Qavg[j] = oldQavg[table[j]];
					parent[j] = oldparent[table[j]];
				}
				Q += Qavg[j];
				Q2 += Qavg[j]*Qavg[j]; 
			}
		}

		table.erase(table.begin(),table.end());
		
		//evaluating average observable and large deviation function
		
		if(rank==0){
		Q = Q/Nw;
		Q2 = Q2/Nw;
		sigma = (Q2-Q*Q)/(i+1)/tint;
		Q = Q/(i+1)/tint;
		phi += log(weightsum/Nw);
		mean.push_back(Q);
		var.push_back(sigma);
		ldf.push_back(phi/(i+1)/tint);
		//compute the multiplicity
		for(j=0;j<Nw;j++){
			m.push_back(parent[j]);
		}
		sort(m.begin(),m.end());
		multiplicity.push_back(distance(m.begin(),unique(m.begin(),m.end())));
		m.erase(m.begin(),m.end());

		if(i%(Ntint/100)==0){
                        printf("%lf completed\n",i*100.0/Ntint);
                }
		}
	}
	
	if(rank==0){	
	//write out the result
	FILE *result,*current;
	result = fopen("basicOutput","w");
	fprintf(result,"t	ldf	mean	var	multiplicity\n");
	for(i=0;i<Ntint;i++){
		fprintf(result,"%lf	%.10lf	%.10lf	%lf	%lf\n",(i+1)*tint,ldf[i],mean[i],var[i],multiplicity[i]*1.0/Nw);
	}
	fclose(result);

        current= fopen("Qcurrent","w");
        for(i=0;i<Nw;i++){
                fprintf(current,"%lf\n",Qavg[i]);
        }
        fclose(current);
	}
/*
	for(i=n;i<Nw;i++){
                if(rank==i/n){
                        MPI_Send(&x[i],1,MPI_DOUBLE,0,i,MPI_COMM_WORLD);
                }
                if(rank==0){
                        MPI_Recv(&x[i],1,MPI_DOUBLE,i/n,i,MPI_COMM_WORLD,&status);
                }
        }
*/
        FILE *restart;
	for(i=0;i<nprocs;i++){
		if(rank==i){
        	restart = fopen("/global/scratch/chloegao/restart_tracer_lambda1=0,lambda2=0","a");
        	for(j=0;j<n;j++){
                	fprintf(restart,"%lf	%lf	%lf	%lf\n",x[j],y[j],vx[j],vy[j]);
		}
		fclose(restart);
		}
		MPI_Barrier(MPI_COMM_WORLD);	
        }
        
	MPI_Finalize();
	return 0;
}
