# Federated ProtoNet (Flower)

Tích hợp mô hình Prototypical Network (PN) đã train vào pipeline **Federated Learning** sử dụng framework [Flower (flwr)](https://flower.dev/).

---

## Cấu trúc thư mục

```
federated_protonet/
├── config.yaml        # Cấu hình cho server, client, model, data, optimizer
├── models.py          # Model wrapper (EpisodicProtoNet) + helpers get/set params
├── data_utils.py      # Xây dựng DataLoader episodic cho từng client
├── flower_client.py   # Flower NumPyClient (train + evaluate local)
├── main.py            # Entry-point: start server hoặc client
└── readme.md          # File này
```

---

## Cách hoạt động

```
┌─────────────────────────────────────────────────────────────┐
│                    Flower Server (FedAvg)                    │
│  1. Broadcast global weights (khởi tạo từ checkpoint PN)    │
│  2. Nhận local weights từ tất cả clients                     │
│  3. Aggregate → cập nhật global weights                      │
│  4. Lặp lại num_rounds vòng                                  │
└────────────────┬──────────────────────────────┬─────────────┘
                 │ gRPC                          │ gRPC
        ┌────────▼───────┐             ┌─────────▼──────┐
        │   Client 0     │             │   Client 1     │
        │  dataset_root  │             │  dataset_root  │
        │   (shard 0)    │             │   (shard 1)    │
        │                │             │                │
        │ fit():         │             │ fit():         │
        │  episodic PN   │             │  episodic PN   │
        │  train loop    │             │  train loop    │
        └────────────────┘             └────────────────┘
```

**Mỗi round:**
1. Server gửi global weights → mỗi client
2. Client `fit()`: set weights → train local `local_epochs` epoch với ProtoNet episodic loss → trả về updated weights + metrics
3. Client `evaluate()`: set weights → chạy validation → trả về val_loss + val_acc
4. Server aggregate bằng **FedAvg** (weighted average theo số episode)

---

## Cài đặt

```bash
pip install flwr>=1.5.0 pyyaml torch torchvision
# Các thư viện project gốc đã có trong requirements.txt
pip install -r requirements.txt
```

---

## Cấu hình (`config.yaml`)

| Section | Key | Mô tả |
|---------|-----|--------|
| `server` | `address` | host:port Flower server |
| `server` | `num_rounds` | Số vòng federated |
| `server` | `min_fit_clients` | Số client tối thiểu để bắt đầu round |
| `client` | `local_epochs` | Số epoch train local mỗi round |
| `pretrained` | path | Checkpoint PN đã train (backbone + proto_head) |
| `data` | `client_dataset_roots` | Danh sách path dữ liệu của từng client |
| `model` | `backbone` | Tên backbone (vd: `resnet12`) |
| `model` | `classifier` | Tên classifier (vd: `proto_head`) |
| `optimizer` | `lr` | Learning rate cho federated fine-tuning |

---

## Chạy

### Bước 1 – Khởi động Server

```bash
# Từ thư mục gốc project
python -m federated_protonet.main \
    --mode server \
    --config federated_protonet/config.yaml \
    --rounds 50
```

### Bước 2 – Khởi động Client(s)

Mở **terminal riêng** cho mỗi client:

```bash
# Client 0 (sử dụng data/client_0/miniImageNet)
python -m federated_protonet.main \
    --mode client \
    --client-id 0 \
    --config federated_protonet/config.yaml

# Client 1 (sử dụng data/client_1/miniImageNet)
python -m federated_protonet.main \
    --mode client \
    --client-id 1 \
    --config federated_protonet/config.yaml
```

### Chạy trên nhiều máy

```bash
# Server machine (IP: 192.168.1.10)
python -m federated_protonet.main --mode server --server-addr 0.0.0.0:8080

# Client machine A
python -m federated_protonet.main --mode client --client-id 0 --server-addr 192.168.1.10:8080

# Client machine B
python -m federated_protonet.main --mode client --client-id 1 --server-addr 192.168.1.10:8080
```

---

## Chuẩn bị dữ liệu cho từng client

Mỗi client cần một thư mục dữ liệu miniImageNet riêng (shard độc lập):

```
data/
├── client_0/
│   └── miniImageNet/
│       ├── train/
│       ├── val/
│       └── test/
└── client_1/
    └── miniImageNet/
        ├── train/
        ├── val/
        └── test/
```

Cập nhật `config.yaml`:
```yaml
data:
  client_dataset_roots:
    - "data/client_0/miniImageNet"
    - "data/client_1/miniImageNet"
```

---

## Warm-start từ checkpoint đã train

Server sẽ tự động load checkpoint và broadcast cho tất cả clients ở round đầu tiên:

```yaml
# config.yaml
pretrained: "checkpoint/miniImageNet_Res12_PN/default/ckpt_epoch_76_top1.pth"
```

Checkpoint này là file `.pth` được save bởi `main.py` (key `"model"` trong dict).

---

## Override qua command line

```bash
python -m federated_protonet.main --mode server \
    --rounds 100 \
    --server-addr 0.0.0.0:9999 \
    --gpu 0
```

---

## Luồng code

```
main.py
  └── run_server()          → fl.server.start_server(FedAvg)
  └── run_client()
        └── ProtoNetClient (flower_client.py)
              ├── get_parameters()   → trả về weights hiện tại
              ├── fit()              → set_params → _train_one_epoch → get_params
              └── evaluate()         → set_params → _evaluate
                    └── models.py
                          ├── EpisodicProtoNet.train_forward()
                          └── EpisodicProtoNet.val_forward()
                                └── architectures/ (backbone + proto_head)
                    └── data_utils.py
                          └── create_torch_dataloader (episodic miniImageNet)
```
