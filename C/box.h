//
// Created by blacksun2 on 19-3-24.
//

#ifndef C_BOX_H
#define C_BOX_H

#include<stdlib.h>

typedef struct {
    float x,y,w,h;
} box;

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a,box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;
        s[i].class = 0;
        s[i].probs = probs;
    }

    // O(classes*total*total)
    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator); //descending according to class confidence
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0; //perform NMS
                }
            }
        }
    }
    free(s);
}

#endif //C_BOX_H
