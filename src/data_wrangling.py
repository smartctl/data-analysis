import pandas as pd
from datetime import datetime
import os, sys, re, time, inspect
from itertools import permutations, combinations

class DataWrangle:
    def __init__(self, mapping_set, y_map_set, logger, dstdir="data/wrangle"):
        self.logger = logger
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting DataWrangle initialization.")
        self.dstdir = dstdir
        if not os.path.exists(self.dstdir):
            os.makedirs(self.dstdir)
        # x_feature dtype definition
        self.mapping_set = mapping_set
        self.dtypes_maps = {}
        self.init_dtypes()
        #
        self.y_map_set = y_map_set
        self.y_label_maps = {}
        self.y_label_yellow = []
        self.y_label_red = []
        self.y_label_red_fatal = []
        self.init_y_label_maps()
        #
        self.df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed DataWrangle initialization.")

    def init_y_label_maps(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Initializing y-label mapping.")
        for entry in self.y_map_set:
            name=entry['name']
            label=entry['label']
            self.y_label_maps[name]=label    
            if label == 'yellow':
                self.y_label_yellow.append(name)
            elif label == 'red':
                self.y_label_red.append(name)
            elif label == 'red_fatal':
                self.y_label_red_fatal.append(name)
            else:
                continue
        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed loading {len(self.y_label_maps.keys())} y-label features.")        

    def init_dtypes(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Initializing dtypes mapping.")
        for entry in self.mapping_set:
            for item in entry['names']:
                self.dtypes_maps[item]=entry['dtype']       
        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed dtypes for {len(self.dtypes_maps.keys())} features.")

    def node_from_colname(self, colname: str):
        x=re.findall(r"^node._",colname)
        if len(x) > 0:
            return x[0][:-1]
        else:
            return ''
    
    def map_colname(self, colname: str):
        node_name = self.node_from_colname(colname)
        if node_name != '':
            return colname.replace(node_name,"node0")
        else:
            return colname

    def convert_to_timestamp(self, colname: str):
        """
        Convert all the column values to proper timestamp
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting timestamp convertion of {self.df.shape[0]} rows.")
        self.df[colname]=self.df[colname].apply(lambda x: datetime.fromtimestamp(time.mktime(time.strptime(x, "%Y%m%d-%H%M%S"))))
        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed timestamp convertion of {self.df.shape[0]} rows.")

    def set_dtypes(self, fatal_if_not_mapped=False):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting dtype conversion.")
        self.logger.debug(f"Working dtypes for DataFrame with shape {self.df.shape}")
        cols = self.df.columns
        for colname in cols:
            mapped_colname = self.map_colname(colname)
            try:
                mapped_dtype = self.dtypes_maps[mapped_colname]
            except:
                # if unknown dtype assume string
                self.logger.warn(f"[{inspect.stack()[0][3]}] Missing dtype map for {mapped_colname}({colname}). Using `string`.")
                if fatal_if_not_mapped:
                    import sys
                    print(f"Forcing exit due to missing dtype mapping")
                    sys.exit()
                mapped_dtype = 'string'
            if mapped_dtype == "datetime64":
                self.convert_to_timestamp(colname)
            self.df[colname]=self.df[colname].astype(mapped_dtype)
        self.logger.debug(f"[{inspect.stack()[0][3]}] Dtype conversion completed.")


    def swap_nodes(self,nodeA, nodeB):
        """
        switch node identifier in columns
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting node swap for {nodeA} and {nodeB}")

        cols_to_rename = {}

        nodeA_cols=self.df.filter(regex=nodeA).columns
        nodeA_cols_fixed=nodeA_cols.str.replace(nodeA,nodeB)
        nodeA_map = {nodeA_cols[i]: nodeA_cols_fixed[i] for i in range(len(nodeA_cols))}

        nodeB_cols=self.df.filter(regex=nodeB).columns
        nodeB_cols_fixed=nodeB_cols.str.replace(nodeB,nodeA)
        nodeB_map = {nodeB_cols[i]: nodeB_cols_fixed[i] for i in range(len(nodeB_cols))}

        cols_to_rename.update(nodeA_map)
        cols_to_rename.update(nodeB_map)

        self.df=self.df.rename(columns=cols_to_rename)

        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed node swap for {nodeA} and {nodeB}")


    def fix_node_ordering(self):
        """
        Validate abstracted node name maps to roles:
            nodes 1-3: master, control_plane, worker (optional)
            nodes 4-6: worker
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting node ordering verification.")

        # for item in self.df.columns:
        #     print(item)

        role_control_plane=['node1', 'node2', 'node3']
        role_other=['node4','node5','node6']
        cp_role_fix=[]
        other_role_fix=[]

        cp_nodes=self.df.filter(regex="node._control_plane").columns.str.replace('_control_plane','')

        for cp in role_control_plane:
            if cp not in cp_nodes:
                cp_role_fix.append(cp)

        if len(cp_role_fix) > 0:
            # we need to find correct node and rename columns
            for other in role_other:
                if other in cp_nodes:
                    other_role_fix.append(other)

            if len(cp_role_fix) != len(other_role_fix):
                self.logger.error(f"[{inspect.stack()[0][3]}] Fatal Error. Requires 3 control-plane nodes.")
                sys.exit(1)

            print (f"Need to switch {cp_role_fix} and {other_role_fix}")
            for idx in range(len(cp_role_fix)):
                nodeA=cp_role_fix.pop()
                nodeB=other_role_fix.pop()
                self.swap_nodes(nodeA, nodeB)            

        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed node ordering verification.")

    def randomize_nodes(self):
        """
        Node name maps to roles:
            nodes 1-3: master, control_plane, worker (optional)
            nodes 4-6: worker
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting node combination.")

        # for item in self.df.columns:
        #     print(item)

        role_control_plane=['node1', 'node2', 'node3']
        role_other=['node4','node5','node6']

        df_combined = self.df.copy()
        for c1,c2 in combinations(role_control_plane, 2):
            self.swap_nodes(c1, c2)
            self.logger.debug(f"[swap_nodes:control-plane] {c1} {c2}")
            for w1,w2 in combinations(role_other, 2):
                self.swap_nodes(w1, w2)
                df_combined=pd.concat([df_combined,self.df], ignore_index=True)
                self.logger.debug(f"[swap_nodes:workers] {w1} {w2}")

        df_combined.reset_index(drop=True, inplace=True)
        self.df = df_combined.copy()

        self.logger.debug(f"[{inspect.stack()[0][3]}] Completed node combination.")

    def feature_name_normalization(self):
        """
        Transform feature name into a canonical string based on mapping
        Examples:
            some_colname_m0_example_com to node1_some_colname
            etcd_object_counts_198_18_111_14 to node3_etcd_object_counts

        Fix control plane attributes that use fqdn or ip at as sufix:
            etcd_object_counts_10_0_153_83
            etcd_network_peer_rtt_10_0_153_83
            etcd_failed_proposal_etcd_ip_10_0_153_83_us_east_2_compute_internal
            etcd_fsync_duration_10_0_153_83

            node1_etcd_failed_proposal_etcd_ip_us_west_1_compute_internal
        Should transform to be similar to:
            node1_etcd_object_counts
            node1_etcd_network_peer_rtt
            node1_etcd_failed_proposal
            node1_etcd_fsync_duration
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting feature name normalization.")

        alias_map={}

        etcd_object_counts_cols=self.df.filter(regex='etcd_object_counts').columns
        node_etcd_ip=etcd_object_counts_cols.str.replace('etcd_object_counts','')

        etcd_failed_proposal_cols=self.df.filter(regex='etcd_failed_proposal').columns
        node_etcd_name=etcd_failed_proposal_cols.str.replace('etcd_failed_proposal','')
        node_etcd_name=node_etcd_name.str.replace('node._','')

        etcd_network_peer_rtt_cols=self.df.filter(regex='etcd_network_peer_rtt').columns
        etcd_fsync_duration_cols=self.df.filter(regex='etcd_fsync_duration').columns

        if len(node_etcd_ip) != 3 or len(node_etcd_name) != 3:
                self.logger.error(f"[{inspect.stack()[0][3]}] Fatal Error. Requires 3 etcd nodes.")
                sys.exit(1)
        else:
            cp_nodes_list=['node1', 'node2', 'node3']
            for idx in range(3):
                node_ip = node_etcd_ip[idx]
                # try:
                #     node_name=list(filter(lambda x: node_ip in x, node_etcd_name.to_list()))[0]
                # except:
                #     assume order in the list is correct
                #     node_name=node_etcd_name[idx]
                # assume order in the list is correct
                node_name=node_etcd_name[idx]
                alias_map[node_name]=cp_nodes_list[idx]
                alias_map[node_ip]=cp_nodes_list[idx]

            rename_map={}
            for item in etcd_object_counts_cols:
                for key in alias_map.keys():
                    if key in item:
                        rename_map[item]=alias_map[key]+"_etcd_object_counts"

            for item in etcd_failed_proposal_cols:
                for key in alias_map.keys():
                    if key in item:
                        rename_map[item]=alias_map[key]+"_etcd_failed_proposal"

            for item in etcd_network_peer_rtt_cols:
                for key in alias_map.keys():
                    if key in item:
                        rename_map[item]=alias_map[key]+"_etcd_network_peer_rtt"

            for item in etcd_fsync_duration_cols:
                for key in alias_map.keys():
                    if key in item:
                        rename_map[item]=alias_map[key]+"_etcd_fsync_duration"
            
            self.df.rename(columns = rename_map, inplace = True)
            self.logger.debug(f"[{inspect.stack()[0][3]}] Completing feature name normalization.")

    def load_dataset(self, fname: str, randomize_nodes=False):
        """
        load and clean raw dataset by:
            filling missing values with 0
            reseting index to avoid an index of 0 for all entries
            normalizing column names to remove cluster specific information
            setting columns data types
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Loading dataset.")
        self.df=pd.read_parquet(fname, engine='pyarrow') # load raw dataset
        self.df.fillna(value=0, inplace=True) # Replace None or NaN with 0
        self.df.reset_index(drop=True, inplace=True) # reset index inplace
        self.fix_node_ordering()
        self.feature_name_normalization()
        self.set_dtypes()
        if not 'source' in self.df.columns:
            self.df['source']=str(fname).split('/')[-1] # embed source file name as attribute
        if randomize_nodes:
            self.randomize_nodes()
        self.logger.debug(f"[{inspect.stack()[0][3]}] Processed {fname} with shape {self.df.shape}")
        self.logger.debug(f"[{inspect.stack()[0][3]}] Dataset loaded.\n{self.df.head()}")

    def load_and_combine_datasets(self, file_names: list):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Loading and combining {len(file_names)} datasets.")
        self.combined_df = pd.DataFrame()
        for fname in file_names:
            self.load_dataset(fname)
            #self.df['source']=str(fname).split('/')[-1] # embed source file name as attribute
            self.combined_df=pd.concat([self.combined_df,self.df], ignore_index=True)
            print(self.combined_df.shape)
        self.combined_df.fillna(value=0, inplace=True) # Replace None or NaN with 0
        self.df=self.combined_df.copy()
        # when combining datasets with different boolean features, missing values are set to 0
        # on prevous step. We need to change booleans from 0 to False
        self.set_dtypes()
        # print(self.df.head())
        # print(self.df.dtypes)
        print(f"{self.combined_df.shape} vs {self.combined_df.shape}")
        self.logger.debug(f"[{inspect.stack()[0][3]}] All datasets loaded and combined.")

    def write_dataset(self, file_name: str):
        self.logger.debug(f"Saving data to {self.dstdir} with name {file_name}.")
        fname=self.dstdir+"/"+file_name
        self.df.to_parquet(fname, compression="snappy")
        self.logger.debug(f"Saved {file_name} DataFrame to {fname}")

