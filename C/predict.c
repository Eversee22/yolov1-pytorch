#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include "box.h"
#include "draw.h"
#include "image.h"

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                     "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//void read(const char* filename, int nparam, int** pparams, float** pprobs, box** pboxes){
//    FILE* fp = fopen(filename,"rb");
//    if(!fp){
//        printf("open %s error\n",filename);
//        exit(0);
//    }
//    float* paramsf = (float*)malloc(sizeof(float)*nparam);
//    int* params = (int*)malloc(sizeof(int)*nparam);
//    fread(paramsf,sizeof(float),nparam,fp);
//    for(int i = 0;i<5;++i)
//        params[i] = (int)paramsf[i];
//    int side = params[2],num=params[3],classes=params[4];
//    //printf("side:%d, num:%d, classes:%d\n",side,num,classes);
//    float** probs =
//    if(probs)
//	    fread(probs,sizeof(float),side*side*num*classes,fp);
//    else
//	    printf("probs malloc failed\n");
//    float* boxes = (float*)malloc(sizeof(float)*side*side*num*4);
//    if(boxes)
//	    fread(boxes,sizeof(float),side*side*num*4,fp);
//    else
//	    printf("boxes malloc failed\n");
//    *pparams = params;
//    *pprobs = probs;
//    *pboxes = boxes;
//    free(paramsf);
//    fclose(fp);
//}
void writeback(const char* filename,float** probs, box* boxes, char* imgname, int side, int num, int classes){
	FILE* fp = fopen(filename,"wb");
	int nlen = strlen(imgname);
	fwrite(&nlen,sizeof(int),1,fp);
	fwrite(&side,sizeof(int),1,fp);
	fwrite(&num,sizeof(int),1,fp);
	fwrite(&classes,sizeof(int),1,fp);
	fwrite(imgname,sizeof(char),nlen,fp);
	for(int i = 0;i<side*side*num;++i)
		fwrite(probs[i],sizeof(float),classes,fp);
	fwrite(boxes,sizeof(box),side*side*num,fp);
	fclose(fp);

}

char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

char* splitstr(char *cfgfile,char sep)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, sep)))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}


int main(int argc, char** argv){
	if(argc==1){
		printf("usage:./predict filename\n");
		exit(0);
	}
	const char* filename = argv[1];
	FILE* fp = fopen(filename,"rb");
	if(!fp){
        printf("open %s error\n",filename);
        exit(0);
    }
	int nparams = 4;
	//float* paramsf = (float*)malloc(nparams*sizeof(float));
	//fread(paramsf,sizeof(float), nparams, fp);
	int* params = (int*)malloc(sizeof(int)*nparams);
	fread(params,sizeof(int), nparams, fp);
//	for(int i=0;i<nparams;++i)
//	    params[i] = (int)paramsf[i];
	//free(paramsf);
	int nlen = params[0], side = params[1], num = params[2], classes = params[3];
	char* imgname = calloc(nlen+1,sizeof(char));
	fread(imgname,sizeof(char),nlen,fp);
	imgname[nlen] = '\0';
	printf("image path:%s,side:%d,num:%d,classes:%d\n",imgname,side,num,classes);
	unsigned int total = side*side*num;
	float** probs = calloc(total,sizeof(float*));
	for(int i=0;i<total;++i){
	    probs[i] = calloc(classes,sizeof(float));
	}
	for(int i=0;i<total;++i)
		fread(probs[i],sizeof(float),classes,fp);
	box* boxes = calloc(total, sizeof(box));
	fread(boxes, sizeof(box),total,fp);
	fclose(fp);
	clock_t since = clock();
	do_nms_sort(boxes,probs,total,classes,0.5);
	printf("do nms sort:%fs\n",(double)(clock()-since)/CLOCKS_PER_SEC);
	//writeback("detsnmsbyc",probs,boxes,imgname,side,num,classes);

    image im = load_image_color(imgname,0,0);
	printf("w:%d,h:%d\n",im.w,im.h);
    image **alphabet = load_alphabet();
    draw_detections(im, total, 0.1, boxes, probs, voc_names, alphabet, 20);

    char* imid = splitstr(imgname,'/');
    char* im_name = malloc(sizeof(char)*(strlen(imid)+8));
    sprintf(im_name,"%s_det",imid);

    save_image(im,im_name);
    show_image(im,im_name);

	free_image(im);
	free(imid);
	free(im_name);
    free(imgname);

    for(int i=0;i<total;++i)
    	free(probs[i]);
    free(probs);
    free(boxes);



}


