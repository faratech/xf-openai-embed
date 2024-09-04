<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        form {
            margin-bottom: 20px;
        }
        input[type="text"] {
            width: 300px;
            padding: 8px;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-bottom: 20px;
        }
        .result h3 {
            margin: 0;
            padding: 0;
        }
        .result p {
            margin: 5px 0;
            padding: 0;
        }
        .result .meta {
            font-size: 12px;
            color: gray;
        }
    </style>
</head>
<body>
    <h1>Search</h1>
    <form method="POST" action="">
        <input type="text" name="query" placeholder="Enter search term" required>
        <input type="submit" value="Search">
    </form>

    <?php
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_POST['query'])) {
        $query = $_POST['query'];
        $max_results = 10;

        $url = 'http://127.0.0.1:8000/faiss/search/';

        $data = array(
            'query' => $query,
            'max_results' => $max_results
        );

        $json_data = json_encode($data);

        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
        curl_setopt($ch, CURLOPT_POSTFIELDS, $json_data);

        $response = curl_exec($ch);
        $http_status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        // Debugging output to see raw response
        echo '<pre>';
        echo "HTTP Status: " . $http_status . "\n";
        print_r($response); // Output raw response
        echo '</pre>';

        if ($http_status == 200) {
            $results = json_decode($response, true);

            // Debugging output to see parsed response
            echo '<pre>';
            print_r($results); // Output parsed JSON
            echo '</pre>';

            if (!empty($results['faiss_results'])) {
                echo '<h2>Search Results:</h2>';
                foreach ($results['faiss_results'] as $result) {
                    echo '<div class="result">';
                    echo '<h3>' . htmlspecialchars($result['thread_title']) . '</h3>';
                    echo '<p>' . htmlspecialchars($result['message']) . '</p>';
                    echo '<p class="meta">Post Date: ' . date('Y-m-d H:i:s', $result['post_date']) . '</p>';
                    echo '</div>';
                }
            } else {
                echo '<p>No results found for "' . htmlspecialchars($query) . '".</p>';
            }
        } else {
            echo '<p>An error occurred while processing the search. Please try again later.</p>';
        }
    }
    ?>
</body>
</html>
