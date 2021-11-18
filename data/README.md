# Weibo Topic Dataset

## Description

We collect millions of user's posts from Sina-Weibo (www.weibo.com), which is a most popular social network platform in China.

On Sina-Weibo, users can post text with topic tags surrounded by `#` , for example: 

> Is Iron Man really dead? `#Avengers: Endgame#` `#Iron Man#`.

This post has two topic tags: `#Avengers: Endgame#` and `#Iron Man#`.

**Note:**

- **For the sake of protecting user privacy, the data has been anonymized by replacing user IDs for each user with a hashing value.**

- **We randomly sampled 5% of the cascades in the full dataset in this folder, for testing the program; the full dataset (968MB, which is excessed the size limit of supplementary material) will be made publicly available upon publication of the paper.**

  Static of the sampled dataset are shown as follows:

  | Properties       | WbRepost | WbTopic   |
  | ---------------- | -------- | --------- |
  | **\# Users**     | 58,766   | 117,713   |
  | **# Repost**     | 114,152  | 319,418   |
  | **# Follow**     | 237,519  | 1,321,766 |
  | **# Cascade**    | 521      | 1,168     |
  | **Max. Cascade** | 5,254    | 7,828     |
  | **Min. Cascade** | 52       | 11        |
  | **Avg. Cascade** | 219      | 273       |

## File Tree

```shell
dataset
├─ README.md
├─ global	# global data directory
│    ├─ repost_relationships.txt	# gloabl following relations of repost dataset
│    └─ topic_relationships.txt	# gloabl following relations of topic dataset
├─ repost_cascades	# repost cascade directory
│    └─ cascade_id.csv	# cascade in csv format
└─ topic_cascades	# topic cascade directory
      └─ cascade_id.csv	# cascade in csv format
```

## Data Format

### Post/Repost Cascade

Each cascade is saved in a csv file with following properties:

|   Property   |         Description          |
| :----------: | :--------------------------: |
|   *origin*   | source post id (be reposted) |
| *origin_uid* |     source post user id      |
|     *id*     |           post id            |
|    *uid*     |           user id            |
|  created_at  |       post/repost time       |

- **The properties of italics is the private user information, which are hashed;**
- Property `created_at` is format as `YYYY-mm-DD HH:MM:SS` ;
- If `origin == id` and `origin_uid == uid` , means this post is original and not reposted.

### Following Relationship

Each dataset has a global following network, format as edge list: each line of `dataname_relationships.txt` is a pair of comma separated user ids $u,v$ , means $u$ is followed by $v$ and $v$ is one of $u$'s fans. 

## Data Collection

 We developed a spider based on huyong's open source WeiboSpider project (https://github.com/nghuyong/WeiboSpider) to collect not only user's posts, reposts and following relationships, but also posts with specific topic.

## Data Processing

### Dataset Building

- We built two datasets: WbRepost and WbTopic:

  - On WbRepost, we focus on user's post and its repost to form a single-source cascade, each cascade is a DAG (Directed Acyclic Graph), and we ignore the relations between different cascade.
  - On WbTopic, we consider the topical relationships between user's posts. Each cascade in this dataset has multiple source nodes which have the same topic tag. So the graph structure of cascade is a forest.

The statistics of each full dataset are shown as follows:

| Properties       | WbRepost  | WbTopic   |
| ---------------- | --------- | --------- |
| **\# Users**     | 887,608   | 2,977,573 |
| **# Repost**     | 2,597,945 | 5,417,189 |
| **# Follow**     | 3,693,057 | 9,071,666 |
| **# Cascade**    | 10,421    | 19,691    |
| **Max. Cascade** | 63,599    | 9,262     |
| **Min. Cascade** | 50        | 10        |
| **Avg. Cascade** | 249       | 275       |

### Filter Out

#### Maintain cascade uniqueness

On WbRepost dataset, each cascade is unique.

On WbTopic dataset, there exists a tiny fraction of topic cascades having the same source nodes. As a result, these cascades are identical and we filter out one of them to make sure each cascade is unique.

#### Filter out short cascades

Prediction on short cascade is meaning-less, so we filtered out short cascades.

The minimal size of cascade can be formulated as: $S*T$ , where $S$ is the number of source nodes (each source node corresponds to a sub-cascade) and $T$ is the threshold of size of each single-source sub-cascade.

- For WbRepost, we filtered out the cascades whose size less than $50$ , so the minimal cascade length is $1*50=50$ ;

- For WbTopic, since each cascade is multi-source ($S\ge2$), **to ensure the average cascade size is at the same level of WbRepost**, we set its filtering threshold to $5$ for each single-source sub-cascade inside each cascade, so the minimal cascade size is $2*5=10$ .

## Cite Us

