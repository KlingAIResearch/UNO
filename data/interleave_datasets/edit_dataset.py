# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
import fastparquet
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset, ParquetNestedIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data


class UnifiedUnoEditIterableDataset(InterleavedBaseIterableDataset, ParquetNestedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, metaquery_vit_transform, 
        data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(
            dataset_name, transform, tokenizer, vit_transform,
            data_dir_list, num_used_data, {},
            local_rank, world_size, num_workers, data_status,
        )
        self.metaquery_vit_transform = metaquery_vit_transform

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'vit_image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def parse_row(self, row):
        row["image_list"] = [row["input_img.bytes"], row["output_img.bytes"]]
        row["instruction_list"] = [row["instruction"]]
        caption = row["recaption"]

        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        for idx in range(start_idx + 1, end_idx + 1):
            instruction = row["instruction_list"][idx-1]
            data = self._add_text(data, instruction, need_loss=False)
            if idx != end_idx:
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                    need_loss=True, 
                    need_vae=True, 
                    need_vit=True,
                )
            else:
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                    need_loss=True, 
                    need_vae=False, 
                    need_vit=False,
                )

        # create vit tensor
        vit_transform_stride = self.metaquery_vit_transform.stride
        vit_image_tensor = self.metaquery_vit_transform(Image.open(io.BytesIO(row["image_list"][-1])))
        height, width = vit_image_tensor.shape[1:]
        # data['num_tokens'] += width * height // vit_transform_stride ** 2

        # create metaquery token sequence
        # prompt = '<|metaquery_im_start|>'
        prompt = ''
        for h in range(height // vit_transform_stride):
            prompt += ''.join(f'<|metaquery_im_{i}|>' for i in range(h * width // vit_transform_stride, (h + 1) * width // vit_transform_stride))

        prompt = self.tokenizer.encode(prompt)
        text_ids = prompt
        data['sequence_plan'].append({
            'type': 'masked_text',
            'enable_cfg': 0,
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        })
        data['num_tokens'] += len(prompt)
        data['text_ids_list'].append(text_ids)

        text_ids = self.tokenizer.encode(caption)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'masked_text2',
                'enable_cfg': 0,
                'loss': 1,
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        
        data['vit_image_tensor_list'].append(vit_image_tensor)

        return data

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, parquet_file_path in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                row_idx = 0
                try:
                    df = fastparquet.ParquetFile(parquet_file_path).to_pandas()
                except Exception as e:
                    print(f'Error {e} in rg#{row_idx}, {parquet_file_path}')
                    continue

                for row_idx, row in enumerate(
                    df.iterrows(), start=row_start_id
                ):
                    try:
                        data = self.parse_row(row[1].to_dict())
                        if len(data) == 0:
                            continue
                        data['data_indexes'] = {
                            "data_indexes": [global_row_group_idx, row_idx],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    except Exception as e:
                        print(f'Error {e} in rg#{row_idx}, {parquet_file_path}')
                        continue
                    yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
