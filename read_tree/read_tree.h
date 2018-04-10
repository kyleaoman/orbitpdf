#ifndef READ_TREE_H
#define READ_TREE_H

#include <stdint.h>

#ifndef EXTRA_HALO_INFO
#define EXTRA_HALO_INFO
#endif

typedef struct halotree halotree;
typedef struct halo_list halo_list;
typedef struct halo_index_key halo_index_key;
typedef struct halo halo;

struct halo {
  float scale;
  int64_t id, num_prog, phantom, pid, upid, mmp;
  halo *desc, *parent, *uparent, *prog, *next_coprog;
  float mvir, orig_mvir, rvir, rs, vrms, scale_of_last_MM,
    vmax, pos[3], vel[3];
  EXTRA_HALO_INFO
};

struct halo_index_key {
  int64_t id;
  int64_t index;
};

struct halo_list {
  halo *halos;
  int64_t num_halos;
  float scale;  
  halo_index_key *halo_lookup;
};

struct halotree {
  halo_list *halo_lists;
  int64_t num_lists;
  int64_t *scale_factor_conv;
  int64_t num_scales;
};

extern struct halotree halo_tree;
extern halo_list all_halos;

halo *lookup_halo_in_list(halo_list *hl, int64_t id);
halo_list *lookup_scale(float scale);
halo_list *find_closest_scale(float scale);
void read_tree(const char *filename);
void delete_tree(void);

#endif /* READ_TREE_H */
