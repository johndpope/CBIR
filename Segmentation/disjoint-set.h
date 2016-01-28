#ifndef DISJOINT_SET
#define DISJOINT_SET

#include <misc.h>
#include <vector>

using namespace std;

// disjoint-set forests using union-by-rank and path compression.

typedef struct {
  int rank;
  int p;
  int size;
  rgb color;
} uni_elt;

rgb rgb_mean(rgb a, rgb b) {
  rgb c;
  c.r = (a.r + b.r)/2;
  c.g = (a.g + b.g)/2;
  c.b = (a.b + b.b)/2;
  return c;
}

class universe {
public:
  universe(int elements, vector<rgb> &colorVect);
  ~universe();
  int find(int x);  
  void join(int x, int y);
  int size(int x) const { return elts[x].size; }
  rgb getColor(int x) const {return elts[x].color;}
  int num_sets() const { return num; }

private:
  uni_elt *elts;
  int num;
};

universe::universe(int elements, vector<rgb> &colorVect) {
  elts = new uni_elt[elements];
  num = elements;
  for (int i = 0; i < elements; i++) {
    elts[i].rank = 0;
    elts[i].size = 1;
    elts[i].p = i;
    elts[i].color = colorVect[i];
  }
}
  
universe::~universe() {
  delete [] elts;
}

int universe::find(int x){
  if(x == elts[x].p)
    return x;
  elts[x].p = this->find(elts[x].p);
  return elts[x].p;
}

void universe::join(int x, int y) {
  if (elts[x].rank > elts[y].rank) {
    elts[y].p = x;
    elts[x].size += elts[y].size;
    elts[x].color = rgb_mean(elts[x].color, elts[y].color);
  } else {
    elts[x].p = y;
    elts[y].size += elts[x].size;
    elts[x].color = rgb_mean(elts[x].color, elts[y].color);
    if (elts[x].rank == elts[y].rank)
      elts[y].rank++;
  }
  num--;
}

#endif
