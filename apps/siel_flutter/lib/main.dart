import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const SielApp());
}

class SielApp extends StatelessWidget {
  const SielApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SIEL',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2F5D50)),
        useMaterial3: true,
      ),
      home: const QueryPage(),
    );
  }
}

class QueryPage extends StatefulWidget {
  const QueryPage({super.key});

  @override
  State<QueryPage> createState() => _QueryPageState();
}

class _QueryPageState extends State<QueryPage> {
  final _controller = TextEditingController();
  bool _loading = false;
  Map<String, dynamic>? _response;
  String? _error;

  Future<void> _ask() async {
    setState(() {
      _loading = true;
      _error = null;
      _response = null;
    });

    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:8787/query'),
        headers: {'content-type': 'application/json'},
        body: jsonEncode({'query': _controller.text, 'lang': 'it'}),
      );
      if (response.statusCode >= 400) {
        throw Exception(response.body);
      }
      setState(() => _response = jsonDecode(response.body));
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SIEL')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 860),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextField(
                  controller: _controller,
                  minLines: 2,
                  maxLines: 4,
                  decoration: const InputDecoration(
                    labelText: 'Domanda',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed: _loading ? null : _ask,
                  icon: _loading
                      ? const SizedBox.square(
                          dimension: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.search),
                  label: const Text('Interroga'),
                ),
                const SizedBox(height: 20),
                if (_error != null)
                  Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                if (_response != null) _ResponsePanel(response: _response!),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ResponsePanel extends StatelessWidget {
  const _ResponsePanel({required this.response});

  final Map<String, dynamic> response;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(response['status']?.toString() ?? 'unknown',
                style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 8),
            Text(response['answer']?.toString() ?? ''),
            const Divider(height: 24),
            Text('Confidence: ${response['confidence']}'),
            Text('Support: ${(response['support_ids'] as List?)?.join(', ') ?? ''}'),
          ],
        ),
      ),
    );
  }
}

