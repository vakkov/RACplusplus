<?php
/**
 * Example script that retrieves document embeddings from Redis and clusters
 * them with racplusplus_rac(). Requires the phpredis extension.
 */

// function fetchEmbeddingsByExportHash(Redis $redis, string $index, string $exportHash): array {
//     $query = sprintf('@export_hash:{%s}', $exportHash);

//     // 1) Count without fetching payloads
//     $countResponse = $redis->rawCommand('FT.SEARCH', $index, $query, 'LIMIT', 0, 0);
//     $total = is_array($countResponse) ? (int) $countResponse[0] : 0;

//     if ($total === 0) {
//         return [[], []];
//     }

//     // 2) Retrieve all ids
//     $searchResponse = $redis->rawCommand('FT.SEARCH', $index, $query, 'LIMIT', 0, $total);
//     // FT.SEARCH reply: [total, key1, [field, value...], key2, ...]
//     $keys = [];
//     for ($i = 1; $i < count($searchResponse); $i += 2) {
//         $keys[] = $searchResponse[$i];
//     }

//     // 3) Pipeline HGET embedding
//     $redis->multi(Redis::PIPELINE);
//     foreach ($keys as $key) {
//         $redis->hGet($key, 'embedding');
//     }
//     $results = $redis->exec();  // embeddings JSON strings

//     return [$results, $keys];
// }


$redisHost = getenv('REDIS_HOST') ?: '172.18.1.167';
$redisPort = (int) (getenv('REDIS_PORT') ?: 6379);
$redisDb   = (int) (getenv('REDIS_DB') ?: 'docs');
$keyPrefix = 'doc:task495_40a01d425d600b4f6b3c3f05cd36f61f:';

$redis = new Redis();
$redis->setOption(Redis::OPT_SERIALIZER, Redis::SERIALIZER_NONE);

if (!$redis->connect($redisHost, $redisPort)) {
    fwrite(STDERR, "Unable to connect to Redis at {$redisHost}:{$redisPort}\n");
    exit(1);
}
#$redis->select($redisDb);

// Collect all keys matching the specified prefix.
$keys = $redis->keys($keyPrefix . '*');
sort($keys, SORT_STRING);

$points = [];
foreach ($keys as $key) {
    $embedding = $redis->hGet($key, 'embedding');
    if ($embedding === false || $embedding === null) {
        continue;
    }

    $floats = unpack('g*', $embedding); // little-endian float32, matches Python
    if ($floats === false) {
        continue;
    }

    $points[] = array_values($floats);
}

if (empty($points)) {
    fwrite(STDERR, "No embeddings with key prefix {$keyPrefix}\n");
    exit(1);
}

$maxMergeDistance = 0.35;
$connectivity = null;
$batchSize = 500;
$noProcessors = 8;
$distanceMetric = 'cosine';

$labels = racplusplus_rac(
    $points, 
    $maxMergeDistance, 
    $connectivity, 
    $batchSize, 
    $noProcessors, 
    $distanceMetric
);
print_r($labels);

$resultsPath = getenv('RAC_RESULTS_PATH');
if ($resultsPath !== false && $resultsPath !== '') {
    $encoded = json_encode($labels, JSON_PRETTY_PRINT);
    if ($encoded === false) {
        fwrite(STDERR, "Failed to encode clustering labels as JSON\n");
        exit(1);
    }
    if (file_put_contents($resultsPath, $encoded . PHP_EOL) === false) {
        fwrite(STDERR, "Unable to write clustering labels to {$resultsPath}\n");
        exit(1);
    }
    fprintf(STDERR, "Wrote clustering results to %s\n", $resultsPath);
}
