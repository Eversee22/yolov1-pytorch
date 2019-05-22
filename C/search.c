#include<stdio.h>

int main(){
	float d;
	unsigned count = 0;
	while(scanf("%f",&d)!=-1){
		if(d>0){
			count++;
			printf("%u,%f ",count,d);
		}
	}
	printf("\n");
}
