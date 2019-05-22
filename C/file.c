#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define DFMT "%f "

typedef float DTYPE;
typedef float* PDTYPE;

void write(char* filename, const PDTYPE d, int n){
    FILE* fp = fopen(filename,"wb");
    fwrite(d,sizeof(DTYPE),n,fp);
    fclose(fp);
}

void read(char* filename,unsigned n){
    FILE* fp = fopen(filename,"rb");
    if(!fp){
        printf("open %s error\n",filename);
        exit(0);
    }
    const int sizeoftype = sizeof(DTYPE);
    PDTYPE d = (PDTYPE)malloc(sizeoftype*n);
    fread(d,sizeoftype,n,fp);
    for(int i=0;i<n;++i){
        printf(DFMT,d[i]);
    }
    printf("\n");
    free(d);
    fclose(fp);
}

void rangen(PDTYPE d,int n,float factor){
    srand(6);
    //printf("%d",RAND_MAX);
    for(int i=0;i<n;++i){
        DTYPE a = (float)rand()/RAND_MAX*factor;
        d[i] = a;
    }
}

void parsearg(int argc,char**argv,int* params,char* filename){
    char arg[16];
    //int params[2]={0};
    for(int i=1;i<argc;++i){
        if(argv[i][0]=='-'){
            strcpy(arg,&argv[i][1]);
            if(strcmp(arg,"m")==0){
                i += 1;
                params[0] = atoi(argv[i]);
            }else if(strcmp(arg,"n")==0){
                i += 1;
                params[1] = atoi(argv[i]);
            }else if(strcmp(arg,"a")==0){
		    i += 1;
		    params[2] = atoi(argv[i]);
	    }
        }else{
		strcpy(filename,argv[i]);
	}
    }
}

void help(){
	printf("usage:./file [-n|-m|-a]\n \
		  -n=number of reading\n \
		  -m=number of generating\n \
		  -a=scale of generating\n \
		     filename\n");
}

int main(int argc,char** argv){
    unsigned m,n;
    int params[3] = {0,10,1};
    char filename[256] = {0};
    if(argc==1){
	help();
	exit(0);
    }
    parsearg(argc,argv,params,filename);
    if(strlen(filename)==0){
	    help();
	    exit(0);
    }
    m = params[0];
    n = params[1];
    int a = params[2];
    //char* filename = "test";
    //int d[11] = {12,15,789,49,20,21,1987,17,25,693,11};
    if(m>0){
	PDTYPE d = (PDTYPE)malloc(sizeof(DTYPE)*(m));
	rangen(d,m,a);
	//d[m-1]=11;
	write(filename,d,m);
	free(d);
    }
    if(n>0){
    	read(filename,n);
    }
    return 0;
}

