    # generate names of tbd amount; name ends at first line break char
    def generate(self, amount_names):
        out = []
        for _ in range(amount_names):
            name = []
            # start always with 0 context for linebreak as first char; forward pass expects shape of (1, 1) to work
            context = torch.zeros((1, 1), dtype=torch.long)
            context = context.to(device)
            while True:
                # context must not be greater than context_len, otherwise mat mul in forward pass does not work; cut max latest context
                context_cut = context[:, -context_len:]
                logits, _ = self(context_cut)
                # grab logits at last timestep
                logits = logits[:, -1, :]
                logits = F.softmax(logits, dim=-1)
                idx = torch.multinomial(logits, num_samples=1, replacement=True).item()
                name.append(itos[idx])
                # end name gen when first linebreak is sampled
                if idx == 0:
                    break
                else:
                    # as long as no linebreak is hit, add last idx to context and sample next char for name
                    context = torch.cat((context, torch.tensor([[idx]], dtype=torch.long, device=device)), dim=1)
            out.append("".join(name))
        return out


# sample from model with amount names
m.generate(50)