#include<stdio.h>
#include<stdlib.h>

typedef struct{
	float x,y,w,h;
} box;
int main(int argc, char** argv){
	if(argc==1)
		exit(0);
	
	//float** probs = calloc(20,sizeof(float*));
	//for(int i = 0;i<20;++i)
	//	probs[i] = calloc(10,sizeof(float));
	box* boxes = calloc(20,sizeof(box));
	FILE* fp = fopen(argv[1],"rb");
	fread(boxes,sizeof(box),20,fp);
	
	//for(int i=0;i<20;++i)
	//	fread(probs[i],sizeof(float),10,fp);
	for(int i = 0;i<20;++i){
		printf("%f,%f,%f,%f\n ",boxes[i].x,boxes[i].y,boxes[i].w,boxes[i].h);
		//printf("\n");
	}
	
	fclose(fp);
}
