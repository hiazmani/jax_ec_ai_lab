export default function handler(req, res) {
    if (req.method === 'GET') {
        const scenarios = [
            {
                id: 'citylearn2020',
                name: 'CityLearn 2020',
                description: '9 households',
                duration: '365 days',
                agents: 9,
            },
            {
                id: 'citylearn2021',
                name: 'CityLearn 2021',
                description: '15 mixed-use buildings',
                duration: '365 days',
                agents: 15,
            },
            // Add more scenarios as needed
        ];
        res.status(200).json(scenarios);
    } else {
        res.setHeader('Allow', ['GET']);
        res.status(405).end(`Method ${req.method} Not Allowed`);
    }
}
