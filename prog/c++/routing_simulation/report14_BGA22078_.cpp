#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cfloat>
#include<random>
#include<deque>
#include<list>
#include<set>
#include<algorithm>
#include<assert.h>

using namespace std;

#define SIMULATION_DURATION 100000 // シミュレーション時間（パケットを発生させる期間）
#define QUEUE_CAPACITY 20          // FIFO待ち行列の容量

#define LEARNING_RATE 0.7
//#define DISCOUNT_RATE 

#define UNIT_TIME 1  // 単位時間（隣接ノード間のパケット転送時間を1とする）
#define NUM_NODES 36 // ネットワーク内のノード数
#define OUTPUT_INTERVAL 10

// #define DEBUG // 動作確認用（コメントアウトを外すと，動作確認用情報が標準出力される
#define SEED 1

enum EVENT {GENERATION, SENDING_FIN, RECEIVING, OUTPUT, END}; // 事象の種類

bool sim_end = false; // シミュレーション終了判定用フラグ

typedef struct packet_data { // パケット用構造体
    int gen_time;     // 発生時刻
    int seq_no;       // シーケンス番号
    int arrival_time; // 現ノード到着時刻
    int prev_node_id; // 前ホップノード
    int dst_node_id;  // （最終の）宛先ノード
    set<int> hist_node_id; // 過去経由ノード履歴
} Packet;

typedef struct event_table { // 事象用構造体
    int   occur_time; // 発生時刻
    int   seq_no;     // シーケンス番号
    EVENT type;       // 事象の種類
    int   node_id;    // 処理するノード    
    Packet *pkt;      // 処理対象パケット
    double info_returned; // パケット転送完了時に受信ノードから送信ノードへ返される情報
                          // （送信ノードでQ値を更新するための値として利用できる）
} EventTable; 

typedef struct neighbor_table { // 各ノードがQ値を管理するテーブル
    int    neighbor_node_id;   // 隣接ノード
    double q_value[NUM_NODES]; // 各宛先ノード（＝配列の要素）に関するQ値
} NeighborTable;

typedef struct node_data { // ノード用データ
    int node_id;                   // ノード番号
    list<NeighborTable> neighbors; // Q値を管理するテーブル
    bool sending;         // パケット送信中フラグ（true: パケット送信中）
    deque<Packet*> queue; // 送信待ちパケットを格納するFIFO待ち行列

    int last_next_node_id;     // 最後にパケットを転送したノード
    int last_dst_node_id;      // 最後に転送したパケットの宛先ノード
    int last_pkt_sojourn_time; // 最後に転送したパケットの自ノード内滞在時間
} Node;

class Simulation { // シミュレーション実行用クラス
private:
    int lambda;      // パケット発生率
    int finish_time; // シミュレーションの（本当の）終了時刻

    deque<EventTable> table; // 事象表
    Node node[NUM_NODES];    // ネットワーク内のノード

    int num_generated_packets_unit = 0; // 単位時間あたりに発生させたパケット数
    
    int current_time = 0; // 現在時刻
    int event_seq_no = 0; // 作成した事象のシーケンス番号
    int pkt_seq_no = 0;   // 発生したパケットのシーケンス番号

    int end_to_end_delay = 0;     // End-to-Endパケット転送遅延
    int end_to_end_delay_tmp = 0;
    int num_pkts_received = 0;    // 宛先ノードに到達したパケット数
    int num_pkts_received_tmp = 0;
    int num_pkts_dropped_bufferoverflow = 0; // FIFO待ち行列のオーバーフローで破棄されたパケット数
    int num_pkts_dropped_loop = 0;
    
    ofstream ofs;
    
public:
    Simulation (int _lambda, int _sim_finish_time) { // コンストラクタ
        lambda = _lambda;
        finish_time = _sim_finish_time;

        for (int i = 0; i < NUM_NODES; i++) { // ノードの初期化
            node[i].node_id = i;
            node[i].sending = false;
        }

        // ネットワークトポロジの設定
        ifstream ifs("topology.txt");
        if (!ifs) {
            cerr << "cannot open file \"topology.txt\"" << endl;
            exit(1);
        }

        string link;
        int one_node, another_node;

        while (getline(ifs, link)) {
            stringstream ss(link);
            ss >> one_node >> another_node;

            NeighborTable neighbor;

            for (int i = 0; i < NUM_NODES; i++) {
                neighbor.q_value[i] = 100; // Q値を初期化
            }
            
            neighbor.neighbor_node_id = another_node;
            node[one_node].neighbors.push_front(neighbor);

            neighbor.neighbor_node_id = one_node;
            node[another_node].neighbors.push_front(neighbor);
        }

        // Q値を管理するテーブルの初期化
        ifstream ifs_routes("routes.txt");
        if (!ifs_routes) {
            cerr << "cannot open file \"routes.txt\"" << endl;
            exit(1);
        }
        
        string routes, dest_nodes_slist, dest_nodes_cons;
        int start_node, dest_node, next_node;

        string::size_type pos_slist = 0, pos_cons = 0;

        while (getline(ifs_routes, routes)) {

            if (routes.substr(0, 1) == "#") continue; // # から始まる行は無視
            
            stringstream ss_routes(routes);
            ss_routes >> start_node >> dest_nodes_slist >> next_node;
#if defined (DEBUG)
            cout << "start_node: " << start_node << endl;
            cout << "  next_node:" << next_node;
#endif
            // 各隣接ノードのイテレータを取得し，後で各宛先ノードに関するQ値を更新
            list<NeighborTable>::iterator p;

            for (p = node[start_node].neighbors.begin(); p != node[start_node].neighbors.end();p++){
                if (p->neighbor_node_id == next_node) break;
            }
            p->q_value[start_node] = 0;
            assert(p != node[start_node].neighbors.end());
#if defined (DEBUG)
            cout << "  dest_node:";
#endif            
            dest_nodes_slist = dest_nodes_slist + ",";
            
            while ((pos_slist = dest_nodes_slist.find(",")) != string::npos) {

                dest_nodes_cons = dest_nodes_slist.substr(0, pos_slist);
                
                if ((pos_cons = dest_nodes_cons.find("-")) != string::npos) {

                    int dest_nodes_start = atoi(dest_nodes_cons.substr(0,pos_cons).c_str());
                    dest_nodes_cons.erase(0, pos_cons + 1);
                    int dest_nodes_end = atoi(dest_nodes_cons.c_str());
#if defined (DEBUG)
                    cout << " " << dest_nodes_start << "-" << dest_nodes_end;
#endif
                    for (int i = dest_nodes_start; i <= dest_nodes_end; i++) {
                        if (i != start_node) {
                            if (i != p->neighbor_node_id) { p->q_value[i] = 10;
                            } else { p->q_value[i] = 5; }
                        }
                    }

                } else {
                    dest_node = atoi(dest_nodes_cons.substr(0, pos_slist).c_str());
#if defined (DEBUG)
                    cout << " " << dest_node;
#endif
                    if (dest_node != start_node) {
                        if (dest_node != p->neighbor_node_id) { p->q_value[dest_node] = 10;
                        } else { p->q_value[dest_node] = 5; }
                    }
                }

                dest_nodes_slist.erase(0, pos_slist + 1);
            }
#if defined (DEBUG)
            cout << endl;
#endif
        }
#if defined (DEBUG)
        for (int i = 0; i < NUM_NODES; i++) {
            show_q_table(i);
        }
#endif
//        ofs.open("packet_transfer_delay.txt", ios::out);
        
        add_event(finish_time, END, 0); // シミュレーション終了事象を事象表に登録

        add_event(OUTPUT_INTERVAL, OUTPUT, 0); // シミュレーション途中の統計情報を出力する事象を登録

        int init_node_id = rand() % NUM_NODES;
        add_event(0, GENERATION, init_node_id); // 最初のパケット発生事象を事象表に登録
    }

    void add_event(int _occur_time, EVENT _type, int _node_id, // 事象を生成し登録
                   Packet *_pkt = NULL, double _info_returned = DBL_MAX) {

        EventTable event;
        event.occur_time = _occur_time;
        event.type = _type;
        event.node_id = _node_id;
        event.seq_no = event_seq_no;
        event_seq_no++;

        event.pkt = _pkt;
        event.info_returned = _info_returned;

        table.push_back(event); // 事象表の末尾に挿入

#if defined (DEBUG)
        cout << "add_event node_id:" << event.node_id << " type:" << event.type 
             << " seq_no:" << event.seq_no << " occur_time:" << event.occur_time << endl;
//        show_event_table();
#endif
    }

    void process_event() { // 事象表の先頭事象を取り出して実行
        // 事象を時刻順に整列
        sort(table.begin(), table.end(), [](EventTable x, EventTable y){
                                             return x.occur_time < y.occur_time;
                                         });
        EventTable &event = table.front(); // 事象表の先頭事象を取得
        current_time = event.occur_time;   // 現在時刻を取り出した事象の発生時刻まで進める
        int node_id = event.node_id;
#if defined (DEBUG)
        show_event_table();        
        cout << "proc_event node_id:" << event.node_id << " type:" << event.type
             << " seq_no:" << event.seq_no << " occur_time:" << event.occur_time << endl;
#endif
        switch(event.type) { // 事象の種類に応じた処理を実行

        case GENERATION: // パケット発生事象の処理
        {
            // シミュレーション時間を過ぎると新規パケットは発生させず
            // 発生済パケットだけ転送する
            if (current_time <= SIMULATION_DURATION) {

                // パケットを生成
                Packet *pkt = new Packet;

                int dst_node_id;    
                pkt->gen_time = current_time;
                pkt_seq_no++;
                pkt->seq_no = pkt_seq_no;
                pkt->arrival_time = current_time;
            
                do {
                    dst_node_id = rand() % NUM_NODES;
                } while (dst_node_id == node_id);

                pkt->dst_node_id = dst_node_id;
      
                int src_node_id = rand() % NUM_NODES;
                num_generated_packets_unit++;

                // 次のパケット発生事象を作成し，事象表に登録
                if (num_generated_packets_unit < lambda) {
                    add_event(current_time, GENERATION, src_node_id);
                } else {
                    add_event(current_time + UNIT_TIME, GENERATION, src_node_id);
                    num_generated_packets_unit = 0;
                }

                // 発生させたパケットの送信処理
                handle_packet(pkt, node_id);
            }
        }

        break;
    
        case RECEIVING: // パケット受信事象の処理
        {
            int dst_node_id  = event.pkt->dst_node_id; // d
            int prev_node_id = event.pkt->prev_node_id; // x
            // cout << "\n(node_id_R)" << node_id << "\n(prev_R)" << prev_node_id << "\n(dst_R)" << dst_node_id << endl;

            if (dst_node_id == node_id) { // 受信パケットの宛先が自ノードの場合

                // パケットを受信し，受信したパケットの結果を測定
#if defined (DEBUG)
                cout << "  RECEIVING" << endl;
#endif
                num_pkts_received_tmp++;
                end_to_end_delay_tmp = end_to_end_delay_tmp + current_time - event.pkt->gen_time;

//                if (current_time >= start_time) {
                num_pkts_received++;
                end_to_end_delay = end_to_end_delay + current_time - event.pkt->gen_time;
//                }
                // cout << "delete pkt " << endl;
                delete event.pkt;
        
            } else { // 受信パケットの宛先が他ノードの場合
                event.pkt->arrival_time = current_time;
                handle_packet(event.pkt, node_id); // 
            }

            list<NeighborTable>::iterator p;

    
            // 前ホップのパケット転送元ノードがQ値を更新するために用いるフィードバック情報を返信
            double q_min = DBL_MAX;

            // 各自フィードバック情報を定義して返信(update)
            double q_value_new;
            double t = min_q_value(node_id, dst_node_id); // y(node_id)からd(dst_node_id)に向かう時のyのとりうる最小q値
            int qx = node[prev_node_id].last_pkt_sojourn_time; // x(prev_node_id)における滞在時間

            // list<NeighborTable>::iterator p;
            for (p = node[prev_node_id].neighbors.begin(); p != node[prev_node_id].neighbors.end(); p++) {
                if (p->neighbor_node_id == node_id){
                    q_value_new = p->q_value[dst_node_id];
                    q_value_new += LEARNING_RATE*(qx+t-p->q_value[dst_node_id]);
                }
            }
            
            // 前ホップノードの送信完了を事象表に格納
            add_event(current_time, SENDING_FIN, prev_node_id, event.pkt, q_value_new);
            // add_event(current_time, SENDING_FIN, prev_node_id, NULL, q_min);
        }
    
        break;

        case SENDING_FIN: // パケット送信完了事象の処理
        {
#if defined (DEBUG)
            cout << "  SENDING_FIN info_returned:" << event.info_returned;
            cout << " last_sojourn_time:" << node[node_id].last_pkt_sojourn_time << endl;
#endif
            // ここでQ値を更新できる(update)
            // ここではフィードバックをもとに受け取った値で更新するだけ，値の計算はRECIEVINGで行う．
            int prev_node_id = event.node_id; // x
            int dst_node_id  = event.pkt->dst_node_id; // d
            int node_id_ = event.pkt->prev_node_id; // y
            double q_value_new = event.info_returned;
            // cout << "\n(node_id_S)" << node_id_ << "\n(prev_S)" << prev_node_id << "\n(dst_S)" << dst_node_id << endl;
            // cout << dst_node_id << endl;
            list<NeighborTable>::iterator p;
            // xのneighborの中からyを探索
            for (p = node[prev_node_id].neighbors.begin(); p != node[prev_node_id].neighbors.end(); p++) { 
                // cout << p->neighbor_node_id << endl;
                if (p->neighbor_node_id == node_id_){ // yの探索
                    p->q_value[dst_node_id] = q_value_new;
                }
            }



            assert(node[node_id].sending);
            node[node_id].sending = false;
#if defined (DEBUG)
            show_q_table(node_id);
#endif
            // FIFO待ち行列に送信待ちパケットがあれば，先頭パケットを送信
            if (node[node_id].queue.empty() != true) {

                Packet *pkt = node[node_id].queue.front();

                int next_node_id = decide_next_node(pkt, node_id);
                if (next_node_id == -1) {
                    num_pkts_dropped_loop++;
                    break;
                }
                node[node_id].last_next_node_id = next_node_id;
                node[node_id].last_dst_node_id = pkt->dst_node_id;
                node[node_id].last_pkt_sojourn_time = current_time - pkt->arrival_time;

                node[node_id].sending = true;

                pkt->prev_node_id = node_id;
                pkt->hist_node_id.insert(node_id);
                // 次ホップノードの受信を事象表に格納（発生時刻は1単位時間後）
                add_event(current_time + UNIT_TIME, RECEIVING, next_node_id, pkt);

                node[node_id].queue.pop_front();
            }
        }
        
        break;

        case OUTPUT:
        {
/*          if (num_pkts_received_tmp > 0) {
                ofs << current_time << "\t" 
                    << (double)end_to_end_delay_tmp / (double)num_pkts_received_tmp << endl;
                end_to_end_delay_tmp = 0;
                num_pkts_received_tmp = 0;
            } */
            add_event(current_time + OUTPUT_INTERVAL, OUTPUT, 0);
        }
        
        break;

        case END: // シミュレーション終了事象の処理
        {
            // 各種結果，統計の算出
            cout << "Average end-to-end packet transfer delay = " 
                 << (double)end_to_end_delay / (double)num_pkts_received << endl;
/* 
            cout << "Packet loss rate = "
                 << 1.0 - (double)num_pkts_received / (double)pkt_seq_no << endl;
            cout << "pkt_seq_no: " << pkt_seq_no << endl;
            cout << "num_pkts_received: " << num_pkts_received << endl;
            cout << "num_pkts_dropped_bufferoverflow: " << num_pkts_dropped_bufferoverflow << endl;
            cout << "num_pkts_dropped_loop: " << num_pkts_dropped_loop << endl;
*/
            sim_end = true;
        }

        break;
    
        default:
            cerr << "unknown event" << endl;
            exit(1);
            break;
        }

        // 上で処理した事象を事象表から削除
        table.pop_front();
    }

#if defined (DEBUG)
    void show_event_table() {
        deque<EventTable>::iterator p;
        cout << endl;
        cout << "event_table" << endl;
            
        for(p = table.begin(); p != table.end(); p++) {
            cout << " node_id:" << p->node_id << " type:" << p->type
                 << " seq_no:" << p->seq_no << " occur_time:" << p->occur_time << endl;
        }
        cout << endl;
    }

    void show_q_table(int node_id) {
        list<NeighborTable>::iterator p;
        cout << endl;
        cout << "Q table node_id:" << node_id << endl;

        for (p = node[node_id].neighbors.begin(); p != node[node_id].neighbors.end(); p++) {
            cout << "  neighbor_node_id:" << p->neighbor_node_id << endl;
            for (int i = 0; i < NUM_NODES; i++) {
                cout << "    q_value[" << i << "]:" << p->q_value[i] << endl;
            }
        }
    }
#endif

    // パケット（_pkt）をノード（_node_id）において処理
    void handle_packet(Packet *_pkt, int _node_id) {
#if defined (DEBUG)
        cout << "handle_packet dst_node_id:" << _pkt->dst_node_id
             << " pkt->seq_no:" << _pkt->seq_no << " gen_time:" << _pkt->gen_time << endl;
#endif
        // 待ち行列に送信待ちパケットが無く，かつ当該ノードがパケット送信中でない場合
        // 本パケットを次ホップノードへ転送する処理を開始
        if (node[_node_id].queue.empty() && node[_node_id].sending == false) {

            int next_node_id = decide_next_node(_pkt, _node_id); // 転送先ノードを決定
            if (next_node_id == -1) { // 転送先が無ければ，パケットを破棄
                num_pkts_dropped_loop++;
                return;
            }
            
            node[_node_id].last_next_node_id = next_node_id;
            node[_node_id].last_dst_node_id  = _pkt->dst_node_id;
            node[_node_id].last_pkt_sojourn_time = current_time - _pkt->arrival_time;

            node[_node_id].sending = true;
            
            _pkt->prev_node_id = _node_id;
            _pkt->hist_node_id.insert(_node_id);
            // 次ホップ転送ノードでのパケット受信事象を事象表に登録（事象発生時刻は1単位時間後）
            add_event(current_time + UNIT_TIME, RECEIVING, next_node_id, _pkt);

        } else { // 待ち行列にパケットがある場合
#if defined (DEBUG)
            cout << "  enqueue" << endl;
#endif
            if (node[_node_id].queue.size() < QUEUE_CAPACITY) { // 待ち行列に空きがある場合

                node[_node_id].queue.push_back(_pkt);

            } else { // 待ち行列に空きが無い場合
                // 結果測定
                num_pkts_dropped_bufferoverflow++;
                delete _pkt;
            }
        }
    }

    // ノード（_node_id）において，パケット（_pkt）の次ホップ転送先ノードを決定
    int decide_next_node(Packet *_pkt, int _node_id) {
#if defined (DEBUG)
        cout << "decide_next_node node_id:" << _node_id 
             << " dst_node_id:" << _pkt->dst_node_id << endl;
        show_q_table(_node_id);
#endif 
        double q_min = DBL_MAX; // 最小のQ値
        int next_node_id = -1;  // 次ホップ転送先ノード
        deque<int> neighbor_node_candidates; // 次ホップ転送先ノードの候補
    
        // Q値が最小の隣接ノードを探索
        list<NeighborTable>::iterator p;
        for (p = node[_node_id].neighbors.begin(); p != node[_node_id].neighbors.end(); p++) {

            // 本パケットの宛先ノードが隣接ノードなら，その隣接ノードへ転送
            if (p->neighbor_node_id == _pkt->dst_node_id) {
#if defined (DEBUG)
                cout << "  next_node_id:" << _pkt->dst_node_id << " (neighbor == dest)" << endl;
#endif
                return (_pkt->dst_node_id);
            }

            // 本パケットをこれまで転送してないノードからQ値が最小のノードを探索
            if (_pkt->hist_node_id.find(p->neighbor_node_id) == _pkt->hist_node_id.end()) {

                if (q_min > p->q_value[_pkt->dst_node_id]) {
                    neighbor_node_candidates.clear();
                    q_min = p->q_value[_pkt->dst_node_id];
                    neighbor_node_candidates.push_front(p->neighbor_node_id);
                }
            }
        }

        if (neighbor_node_candidates.size() > 0) {
            // Q値が最小のノードが複数ある場合，それらから等確率でランダムに決定
            int next_node_el = rand() % neighbor_node_candidates.size();
            next_node_id = neighbor_node_candidates[next_node_el];
        }
#if defined (DEBUG)
        cout << "  next_node_id:" << next_node_id << endl;
#endif
        return next_node_id;
    }

    int min_q_value(int node_id, int dst_node_id) {
        double q_min = DBL_MAX; // 最小のQ値

        list<NeighborTable>::iterator p;
        for (p = node[node_id].neighbors.begin(); p != node[node_id].neighbors.end(); p++) {
            for (int i = 0; i < NUM_NODES; i++) {
                if (p->neighbor_node_id == dst_node_id && p->q_value[i] < q_min){
                    q_min = p->q_value[i];
                }
            }
        }
        return q_min;
    }
};

int main(int argc, char **argv) {
    
    if (argc != 2) {
        cout << "error : invalid arguments" << endl;
        exit(1);
    }

    int lambda = atoi(argv[1]); // パケット発生率
    srand(SEED);

    // 時刻SIMULATION_DURATION にパケット発生を終了させた後，
    // ネットワーク内に残存しているパケットを宛先ノードまで到達させる
    // （+1000は到達のために確保する追加時間）
    int sim_finish_time = SIMULATION_DURATION + 1000; // シミュレーションの（本当の）終了時刻
    
    Simulation obj(lambda, /* record_start_time, */ sim_finish_time);

    while (sim_end == false) {
        obj.process_event();
    }

    return 0;
}
