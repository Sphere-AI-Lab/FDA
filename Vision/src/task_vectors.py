import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    # print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]

            for key in other.vector:
                if key not in self.vector:
                    # print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __mul__(self, scalar):
        """Scale the task vector by a scalar."""
        if isinstance(scalar, (int, float)):  # Ensure it's a scalar (int or float)
            with torch.no_grad():
                new_vector = {key: self.vector[key] * scalar for key in self.vector}
            return TaskVector(vector=new_vector)
        else:
            raise NotImplementedError("Multiplication with non-scalar types is not supported.")

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def __truediv__(self, scalar):
        """Divide the task vector by a scalar."""
        if isinstance(scalar, (int, float)):  # Ensure it's a scalar (int or float)
            with torch.no_grad():
                new_vector = {key: self.vector[key] / scalar for key in self.vector}
            return TaskVector(vector=new_vector)
        else:
            raise NotImplementedError("Division with non-scalar types is not supported.")

    def norm(self):
        """Calculate the norm of the entire task vector."""

        total_norm = 0.0
        with torch.no_grad():
            for key in self.vector:
                total_norm += torch.norm(self.vector[key].flatten()).item()**2
        total_norm **= 0.5
        return total_norm

    def dot(self, other):
        """Calculate the dot product between two task vectors."""
        inner_product = 0.0
        with torch.no_grad():
            for key in self.vector:
                if key not in other.vector:
                    # print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                inner_product += torch.dot(self.vector[key].flatten(),
                                           other.vector[key].flatten().to(self.vector[key].device))
        return inner_product

    def quantile(self, q, abs=True):
        """Calculate the q-th quantile of the task vector values."""
        if not (0 <= q <= 1):
            raise ValueError("Quantile must be between 0 and 1.")

        with torch.no_grad():
            all_values = torch.cat([value.flatten() for value in self.vector.values()])
            if abs:
                all_values = torch.abs(all_values)

            sorted_tensor = torch.sort(all_values).values
            idx = int(q * len(sorted_tensor))
            quantile_value = sorted_tensor[idx].item()

        return quantile_value

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Calculate the cosine similarity between two task vectors."""
        if vec1.norm() == 0 or vec2.norm() == 0:
            raise ValueError("Cosine similarity is not defined when one of the vectors has zero magnitude.")
        return vec1.dot(vec2) / (vec1.norm() * vec2.norm())

    def drop_small(self, ratio=0.9):
        """Only Keep the top-k values in the task vector, and set others to 0, where k = (1-ratio) * total_keys."""
        with torch.no_grad():
            all_values = torch.cat([value.flatten() for value in self.vector.values()])
            sorted_tensor = torch.sort(torch.abs(all_values)).values
            idx = int(ratio * len(sorted_tensor))
            threshold = sorted_tensor[idx].item()

            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * (torch.abs(self.vector[key]) > threshold).float()
        return TaskVector(vector=new_vector)
    
    def drop_random(self, ratio=0.9, rescale=True):
        """Randomly set a fraction of the values in the task vector to 0."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                mask = torch.rand_like(self.vector[key]) > ratio
                new_vector[key] = self.vector[key] * mask.float()
                if rescale:
                    new_vector[key] *= 1.0 / (1 - ratio)
        return TaskVector(vector=new_vector)
    
    def drop_large(self, ratio=0.9, rescale=True):
        """Drop the ratio fraction of the largest values in the task vector, and optionally rescale the remaining values back to the scale of the original ones."""
        with torch.no_grad():
            all_values = torch.cat([value.flatten() for value in self.vector.values()])
            sorted_tensor = torch.sort(torch.abs(all_values)).values
            idx = int((1 - ratio) * len(sorted_tensor))
            threshold = sorted_tensor[idx].item()

            new_vector = {}
            for key in self.vector:
                norm = torch.norm(self.vector[key].flatten())
                mask = torch.abs(self.vector[key]) < threshold
                new_vector[key] = self.vector[key] * mask.float()
                if rescale:
                    new_vector[key] *= norm / torch.norm(new_vector[key].flatten())
        return TaskVector(vector=new_vector)


    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        device = self.vector[list(self.vector.keys())[0]].device

        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint, map_location=device)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    # print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def save(self, path):
        """Saves the task vector to a file."""
        torch.save(self.vector, path)

    def to(self, device):
        """Move the task vector to a device."""
        with torch.no_grad():
            new_vector = {key: value.to(device) for key, value in self.vector.items()}
        return TaskVector(vector=new_vector)

    @staticmethod
    def load(path, device=None):
        """Loads the task vector from a file and returns a TaskVector object."""
        vector = torch.load(path)
        # vector = {key: value for key, value in vector.items() if 'visual' in key}
        if device:
            vector = {key: value.to(device) for key, value in vector.items()}
        return TaskVector(vector=vector)

    @staticmethod
    def load_from_model(path, pretrained_path, device=None):
        """Loads a model from a file and returns a TaskVector object."""
        model = torch.load(path)
        pretrained_model = torch.load(pretrained_path)
        if device:
            model = model.to(device)
            pretrained_model = pretrained_model.to(device)
        vector = {}
        for key in model.state_dict():
            vector[key] = model.state_dict()[key] - pretrained_model.state_dict()[key]
        return TaskVector(vector=vector)
