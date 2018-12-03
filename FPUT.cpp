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
/*
double subtract_max(vector<double> weight){
        int k;
        double logsum = 0.0;
        double max = 0.0;

        for(k=0;k<NWALKER;k++){
                if(weight[k]>max)       max=weight[k];
        }
        for(k=0;k<NWALKER;k++){
                logsum+=exp(weight[k]-max);
        }
        logsum = log(logsum)+max;
        return logsum;
}
*/
vector<double> getforce(const vector<double> x, int N, double alpha, double beta){
	int i;
	vector<double> f(N);
	double xstart = -1*alpha;
	double xend = N*alpha;

	for(i=1;i<(N-1);i++){
		f[i] = (-2*x[i]+x[i+1]+x[i-1])-beta*(x[i]-x[i-1]-alpha)*(x[i]-x[i-1]-alpha)*(x[i]-x[i-1]-alpha)+beta*(x[i+1]-x[i]-alpha)*(x[i+1]-x[i]-alpha)*(x[i+1]-x[i]-alpha);
	}
	f[0] = (x[1]+xstart-2*x[0])+beta*(x[1]-x[0]-alpha)*(x[1]-x[0]-alpha)*(x[1]-x[0]-alpha)-beta*(x[0]-xstart-alpha)*(x[0]-xstart-alpha)*(x[0]-xstart-alpha);
	f[N-1] = (x[N-2]+xend-2*x[N-1])+beta*(xend-x[N-1]-alpha)*(xend-x[N-1]-alpha)*(xend-x[N-1]-alpha)-beta*(x[N-1]-x[N-2]-alpha)*(x[N-1]-x[N-2]-alpha)*(x[N-1]-x[N-2]-alpha);
	return f;
}

// this function takes the initial value x&v and modify it
// and returns the cumulative current Q within tint 
double propagation(vector<double>& x, vector<double>& v, double& q1, const vector<double> mass, int N, double tint, gsl_rng* rng)
{
        double h = 0.005;
        int Nstep = tint/h;
        int i,j;
	double alpha = 2.0;
	double beta = 1.0;
	double gamma = 0.8;
	double dt = 1.0;
	int interval = dt/h;
	// Th is the temperature of the first atom while Tl is the last
	double T0 = 0.1;
	double det = 0.0;
	double Tl = T0-det;
	double Th = T0+det;
	double dummy,temp;
	double q2 = 0.0;
	double voldstart,foldstart,voldend,foldend;
	vector<double> f(N);

	f = getforce(x,N,alpha,beta);
         
	for(i=0;i<Nstep;i++){
		voldstart = v[0];
		foldstart = f[0];
		voldend = v[N-1];
		foldend = f[N-1];

		//update half-step velocity
		for(j=0;j<N;j++){
			v[j] += h*f[j]/2/mass[j];
		}
					
		//update positions
		for(j=0;j<N;j++){
			x[j] += h*v[j];
		}
		
		//update full-step velocity
		f = getforce(x,N,alpha,beta);
		for(j=0;j<N;j++){
			v[j] += h*f[j]/2/mass[j];
		}
		
		//Anderson thermostat
		if((i%interval)==0){
			dummy = gsl_rng_uniform(rng);
			if(dummy<(gamma*dt)){
				temp = gsl_ran_gaussian_ziggurat(rng,1);
				v[0] = sqrt(Th/mass[0])*temp;
				q2 += 0.25*T0*temp*temp;
			}
			dummy = gsl_rng_uniform(rng);
                        if(dummy<(gamma*dt)){
                                temp = gsl_ran_gaussian_ziggurat(rng,1);
                                v[N-1] = sqrt(Tl/mass[N-1])*temp;
                                q2 -= 0.25*T0*temp*temp;
                        }
		}
		
		//compute the flux
		q1 += 0.5*(mass[0]*0.5*(v[0]*v[0]-voldstart*voldstart)-foldstart*voldstart*h)-0.5*(mass[N-1]*0.5*(v[N-1]*v[N-1]-voldend*voldend)-foldend*voldend*h);
	}
        return q2;
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

	//parameters in the FPU chain model
	int N = 20;
	double Ml = 1.0;
	double Mh = 20.0;
	vector<double> mass(N), xinit(N), vinit(N);
	int i,j,k,s,index;

	for(i=0;i<N;i++){
		mass[i] = Ml + i*(Mh-Ml)/(N-1);
	}
	
	FILE *init;
 	if ( ( init = fopen( "eq", "r" ) ) == NULL){
   		printf ("initial data could not be opened\n");}
 	else {
      		for(i=0;i<N;i++){
		fscanf(init, "%lf	%lf", &xinit[i],&vinit[i]);
     		}
   	 }
   	rewind(init);
   	fclose(init);
 	
	//parameters for CANSS
	int Nw = 57600;
	int n = Nw/nprocs;
	double lambda1 = 0.0;
	double lambda2 = 0.0;
	double tint = 200.0, tobs = 200000.0;
	int Ntint = tobs/tint;
	double weightsum = 0.0;
	int walkersum = 0;
	vector<double> globalq(Nw),localq(n);
	vector<double> globalweight(Nw),localweight(n);
	double q1,q2,Q,Q2,localQ,localQ2,sigma;
	double phi = 0.0;

	vector< vector<double> > x(Nw, vector<double>(N));
	vector< vector<double> > v(Nw, vector<double>(N));
	vector< vector<double> > newx(Nw, vector<double>(N));
	vector< vector<double> > newv(Nw, vector<double>(N));
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
	for(i=0;i<Nw;i++){
		x[i] = xinit;
		v[i] = vinit;
		parent[i] = i;
		Qavg[i] = 0.0;
	}
	
	for(i=0;i<Ntint;i++){
		walkersum = 0;
		weightsum = 0.0;

		for(j=0;j<n;j++){
			q1 = 0.0;
			index = rank*n+j;
			q2 = propagation(x[index],v[index],q1,mass,N,tint,rng);
			localq[j] = q1;
			localweight[j] = lambda1*q1+lambda2*q2;
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
			if((table[j]/n)!=(j/n)){
			if(rank==table[j]/n){
				MPI_Send(&x[table[j]][0],N,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
				MPI_Send(&v[table[j]][0],N,MPI_DOUBLE,j/n,j,MPI_COMM_WORLD);
			}
			if(rank==j/n){
				MPI_Recv(&newx[j][0],N,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);
				MPI_Recv(&newv[j][0],N,MPI_DOUBLE,table[j]/n,j,MPI_COMM_WORLD,&status);
				
			}
			}
			else{
				if(rank==j/n){
					newx[j] = x[table[j]];
					newv[j] = v[table[j]];
				}
			}
			}
		}

		// Search in the lookup table for replacing walkers	
		for(j=0;j<n;j++){
			index = rank*n+j;
			if(table[index]!=index){
				x[index] = newx[index];
				v[index] = newv[index];
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
/*
		if(i%(Ntint/100)==0){
                        printf("%lf completed\n",i*100.0/Ntint);
                }
*/
		}
	}
	
	if(rank==0){	
	//write out the result
	FILE *result,*current;
	result = fopen("basicOutput","w");
	fprintf(result,"t	ldf	mean	var	multiplicity\n");
	for(i=0;i<Ntint;i++){
		fprintf(result,"%lf	%.10lf	%.10lf	%.10lf	%lf\n",(i+1)*tint,ldf[i],mean[i],var[i],multiplicity[i]*1.0/Nw);
	}
	fclose(result);

        current= fopen("Qcurrent","w");
        for(i=0;i<Nw;i++){
                fprintf(current,"%lf\n",Qavg[i]);
        }
        fclose(current);
	}

	for(i=n;i<Nw;i++){
                if(rank==i/n){
                        MPI_Send(&x[i][0],N,MPI_DOUBLE,0,i,MPI_COMM_WORLD);
                        MPI_Send(&v[i][0],N,MPI_DOUBLE,0,i,MPI_COMM_WORLD);
                }
                if(rank==0){
                        MPI_Recv(&x[i][0],N,MPI_DOUBLE,i/n,i,MPI_COMM_WORLD,&status);
                        MPI_Recv(&v[i][0],N,MPI_DOUBLE,i/n,i,MPI_COMM_WORLD,&status);
                }
        }

        if(rank==0){
        FILE *restart;
        restart = fopen("/global/scratch/chloegao/restart_N=20_2dcanss_JQ1_Anderson_lambda1=0,lambda2=0","w");
        for(i=0;i<Nw;i++){
                for(j=0;j<N;j++) fprintf(restart,"%lf   %lf\n",x[i][j],v[i][j]);
        }
        fclose(restart);
        }

	MPI_Finalize();
	return 0;
}
